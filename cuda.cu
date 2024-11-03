#include "wukong.h"

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
static size_t cublaslt_workspace_size = 32 * 1024 * 1024;
static void* cublaslt_workspace = NULL;
static cublasComputeType_t cublas_compute_type;
static cublasLtHandle_t cublaslt_handle;
static cublasHandle_t cublas_handle;
static cudnnHandle_t cudnn_handle;
static int cuda_arch_major = 0;
static int cuda_arch_minor = 0;
static int cuda_num_SMs = 0; // for persistent threads where we want 1 threadblock per SM
static int cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU
static int cuda_threads_per_block = 0;
static int cuda_warp_size = 0; // warp size of the GPU

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Handles both scaling of attention scores and softmax computation with causal masking
// inp/out shape: (B, NH, T, T)
__global__ void scaled_softmax_kernel(float* out, const float* inp, int B, int NH, int T, float scale) {
    extern __shared__ float shared[];
    int batch_idx = blockIdx.x / (NH * T); // batch index
    int head_idx = (blockIdx.x / T) % NH;  // head index
    int row_idx = blockIdx.x % T;          // row index within the attention matrix
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;         // warp index within a block
    int laneId = threadIdx.x % 32;         // thread index within a warp
    int warpsPerBlock = blockDim.x / 32;

    // shared memory layout: first half for max values, second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // calculate base index for this thread block's row
    int row_start = (batch_idx * NH * T * T) + (head_idx * T * T) + (row_idx * T);
    const float* x = inp + row_start;

    // Step 1: Find maximum while applying scale and causal mask
    float maxval = -INFINITY;
    for (int i = tid; i < T; i += blockDim.x) {
        float val = (i <= row_idx) ? x[i] * scale : -INFINITY;
        maxval = fmaxf(maxval, val);
    }

    // warp-level reduction for maxval
    maxval = warpReduceMax(maxval);

    // write per-warp maxval to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // final reduction for maxval across warps
    if (tid == 0) {
        float val = maxvals[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();

    // broadcast max to all threads
    float offset = maxvals[0];

    // Step 2: Compute exp(x - max) while respecting causal mask
    float sumval = 0.0f;
    for (int i = tid; i < T; i += blockDim.x) {
        float val = (i <= row_idx) ? expf(x[i] * scale - offset) : 0.0f;
        out[row_start + i] = val;  // store intermediate result
        sumval += val;
    }

    // warp-level reduction for sum
    sumval = warpReduceSum(sumval);

    // write per-warp sum to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // final reduction for sum across warps
    if (tid == 0) {
        float val = sumvals[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();

    // Step 3: Normalize by sum
    float sum = sumvals[0];
    float inv_sum = 1.0f / sum;

    // write final normalized values
    for (int i = tid; i < T; i += blockDim.x) {
        if (i <= row_idx) {
            out[row_start + i] *= inv_sum;
        } else {
            out[row_start + i] = 0.0f;
        }
    }
}

__global__ void softmax_kernel(float* output, const float* input, int row, int col) {
    extern __shared__ float shared_mem[];
    float* row_max = shared_mem;                    // First part of shared memory for max values
    float* row_sum = &shared_mem[blockDim.x / 32];  // Second part for sum values

    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int warps_per_block = blockDim.x / 32;
    int row_idx = blockIdx.x;

    if (row_idx >= row) return;

    // Step 1: Find maximum value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < col; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input[row_idx * col + i]);
    }

    // Warp-level reduction for max
    thread_max = warpReduceMax(thread_max);

    // Store per-warp results
    if (lane_id == 0) {
        row_max[warp_id] = thread_max;
    }
    __syncthreads();

    // Final reduction for max across warps
    if (tid == 0) {
        float max_val = row_max[0];
        for (int i = 1; i < warps_per_block; i++) {
            max_val = fmaxf(max_val, row_max[i]);
        }
        row_max[0] = max_val;
    }
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    float max_val = row_max[0];
    float thread_sum = 0.0f;

    for (int i = tid; i < col; i += blockDim.x) {
        float val = expf(input[row_idx * col + i] - max_val);
        output[row_idx * col + i] = val;  // Store intermediate result
        thread_sum += val;
    }

    // Warp-level reduction for sum
    thread_sum = warpReduceSum(thread_sum);

    // Store per-warp sums
    if (lane_id == 0) {
        row_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final reduction for sum across warps
    if (tid == 0) {
        float sum = row_sum[0];
        for (int i = 1; i < warps_per_block; i++) {
            sum += row_sum[i];
        }
        row_sum[0] = sum;
    }
    __syncthreads();

    // Step 3: Normalize by sum
    float inv_sum = 1.0f / row_sum[0];
    for (int i = tid; i < col; i += blockDim.x) {
        output[row_idx * col + i] *= inv_sum;
    }
}

__global__ void gqa_permute_kernel(float* q, float* k, float* v, const float* inp,
                                 int B, int N, int qNH, int kvNH, int HS)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gNH = qNH / kvNH; // number of query heads per key/value head

    // Handle Q permute: (B, N, qNH, HS) -> (B, gNH, kvNH, N, HS)
    if (idx < B * qNH * N * HS) {
        int b = idx / (qNH * N * HS);
        int rest = idx % (qNH * N * HS);
        int qh = rest / (N * HS);
        rest = rest % (N * HS);
        int n = rest / HS;
        int hs = rest % HS;

        // Calculate group and kv head indices
        int g = qh / kvNH;  // group index
        int kh = qh % kvNH; // corresponding kv head index

        // Input index
        int inp_idx = (b * N * (qNH + 2 * kvNH) * HS) +
                     (n * (qNH + 2 * kvNH) * HS) +
                     (qh * HS) + hs;

        // Output index for Q
        int q_idx = ((b * gNH * kvNH * N * HS) +
                    (g * kvNH * N * HS) +
                    (kh * N * HS) +
                    (n * HS) + hs);

        q[q_idx] = inp[inp_idx];
    }

    // Handle K,V permute: (B, N, kvNH, HS) -> (B, kvNH, N, HS)
    if (idx < B * kvNH * N * HS) {
        int b = idx / (kvNH * N * HS);
        int rest = idx % (kvNH * N * HS);
        int kh = rest / (N * HS);
        rest = rest % (N * HS);
        int n = rest / HS;
        int hs = rest % HS;

        // Input indices
        int k_inp_idx = (b * N * (qNH + 2 * kvNH) * HS) +
                       (n * (qNH + 2 * kvNH) * HS) +
                       (qNH * HS) +
                       (kh * HS) + hs;

        int v_inp_idx = k_inp_idx + (kvNH * HS);

        // Output indices
        int kv_idx = (b * kvNH * N * HS) +
                    (kh * N * HS) +
                    (n * HS) + hs;

        k[kv_idx] = inp[k_inp_idx];
        v[kv_idx] = inp[v_inp_idx];
    }
}

// Transform from (B, gNH, kvNH, N, HS) to (B, N, qNH, HS)
__global__ void gqa_unpermute_kernel(const float* inp, float* out, int B, int N, int qNH, int kvNH, int HS)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gNH = qNH / kvNH;

    if (idx < B * qNH * N * HS) {
        int b = idx / (qNH * N * HS);
        int rest = idx % (qNH * N * HS);
        int qh = rest / (N * HS);
        rest = rest % (N * HS);
        int n = rest / HS;
        int hs = rest % HS;

        // Calculate group and kv head indices
        int g = qh / kvNH;
        int kh = qh % kvNH;

        // Input index
        int inp_idx = (b * gNH * kvNH * N * HS) +
                     (g * kvNH * N * HS) +
                     (kh * N * HS) +
                     (n * HS) + hs;

        // Output index
        int out_idx = (b * N * qNH * HS) +
                     (n * qNH * HS) +
                     (qh * HS) + hs;

        out[out_idx] = inp[inp_idx];
    }
}

extern "C" {
void cuda_init(void)
{
    srand(0);   // determinism

    // set up the device
    int deviceIdx = 0;
    cuda_check(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_num_SMs = deviceProp.multiProcessorCount;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;
    cuda_threads_per_block = deviceProp.maxThreadsPerBlock;
    cuda_warp_size = deviceProp.warpSize;
    printf("CUDA device: %s, major %d, minor %d, num_SMs: %d, threads_per_SM: %d, threads_per_block: %d, warp_size: %d\n",
           deviceProp.name, cuda_arch_major, cuda_arch_minor, cuda_num_SMs, cuda_threads_per_SM, cuda_threads_per_block, cuda_warp_size);

    cudnn_check(cudnnCreate(&cudnn_handle));
    cublas_check(cublasCreate(&cublas_handle));
    cublas_check(cublasLtCreate(&cublaslt_handle));
    cuda_check(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}

void cuda_fini(void)
{
    cuda_check(cudaFree(cublaslt_workspace));
    cublas_check(cublasLtDestroy(cublaslt_handle));
    cudnn_check(cudnnDestroy(cudnn_handle));
}


void* cuda_malloc(size_t size)
{
    void *ptr;
    cuda_check(cudaMalloc(&ptr, size));
    return ptr;
}

void cuda_free(void* ptr)
{
    cuda_check(cudaFree(ptr));
}

void cuda_to_device(void* dst, void* src, size_t size)
{
    cuda_check(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void cuda_to_host(void* dst, void* src, size_t size)
{
    cuda_check(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

/*
 * Fused matrix multiplication with optional bias addition: out = inp @ weight + bias
 *
 * @param out: output matrix(row, oc)
 * @param inp: input matrix(row, column)
 * @param weight: weight matrix(column, oc)
 * @param bias: optional bias vector(oc) (can be NULL)
 * @param row: input row size
 * @param column: input column size
 * @param oc: output column size
 */
void cuda_matmul(void *out, const void *inp, const void *weight, const void *bias,
            int row, int column, int oc)
{
    int res;
    bool has_bias = (bias != nullptr);
    bool has_gelu = false; /* TODO: Fuse GELU */
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulPreference_t pref;
    cublasLtMatrixLayout_t inp_layout, weight_layout, out_layout, bias_layout;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasOperation_t notrans = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;

    /*
     * Cuda is colum-major, for row-major Array, if we want to get: out = inp @ weight, 'out' should be 'out.T'.
     * Mathematically, out.T = weight.T @ inp.T. Since cuda is colum-major, 'weight.T' should be weight, 'inp.T' should be inp.
     * so calculating out.T = weight & inp.
     */
    cublas_check(cublasLtMatmulDescCreate(&desc, cublas_compute_type, CUDA_R_32F));
    cublas_check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &notrans, sizeof(notrans)));
    cublas_check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &notrans, sizeof(notrans)));
    cublas_check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    cublas_check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    cublas_check(cublasLtMatrixLayoutCreate(&weight_layout, CUDA_R_32F, oc, column, oc));
    cublas_check(cublasLtMatrixLayoutCreate(&inp_layout, CUDA_R_32F, column, row, column));
    cublas_check(cublasLtMatrixLayoutCreate(&out_layout, CUDA_R_32F, oc, row, oc));
    cublas_check(cublasLtMatrixLayoutCreate(&bias_layout, CUDA_R_32F, oc, 1, oc));


    if (has_bias && (uintptr_t)bias % 16 != 0)
        panic("bias must be aligned to 16 bytes");

    cublas_check(cublasLtMatmulPreferenceCreate(&pref));
    cublas_check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    cublas_check(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, desc, weight_layout, inp_layout, out_layout,
                out_layout, pref, 1, &heuristic, &res));
    if (res == 0)
        panic("No algorithm found: row=%d, column=%d, oc=%d, has_bias=%d, has_gelu=%d",
              row, column, oc, has_bias, has_gelu);

    const float alpha = 1.0f, beta = 0.0f;
    cublas_check(cublasLtMatmul(cublaslt_handle, desc, &alpha, weight, weight_layout, inp, inp_layout, &beta,
                out, out_layout, out, out_layout, &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, 0));

    cublas_check(cublasLtMatmulPreferenceDestroy(pref));
    cublas_check(cublasLtMatmulDescDestroy(desc));
    cublas_check(cublasLtMatrixLayoutDestroy(weight_layout));
    cublas_check(cublasLtMatrixLayoutDestroy(inp_layout));
    cublas_check(cublasLtMatrixLayoutDestroy(out_layout));
    cublas_check(cublasLtMatrixLayoutDestroy(bias_layout));
}

/*
 * Row-wise cuda_softmax
 * @param output: shape (row, column)
 * @param input: shape (row, column)
 * @param row: row size
 * @param col: column size
 */
void cuda_softmax(void* output, void* input, int row, int col)
{
    const int block_size = 256;
    const int shared_mem_size = (2 * (block_size / 32)) * sizeof(float); // Space for max and sum values
    softmax_kernel<<<row, block_size, shared_mem_size>>>((float *)output, (const float *)input, row, col);
    cuda_check(cudaGetLastError());
}

/*
 * Vanilla multi-head attention implementation
 *
 * attention = softmax(Q@K^T/sqrt(HS)) @ V
 *
 * @param out: output matrix(batch, row, col)
 * @param inp: input matrix(batch, row, 3 * col) (Q, K, V) concatenated along the last dimension
 * @param batch: batch size
 * @param row: row size
 * @param NH: number of heads
 * @param HS: head size
 * @attention col = NH * HS
 */
void cuda_mha_attention(void *out, const void *inp, int batch, int row, int NH, int HS)
{
    return cuda_gqa_attention(out, inp, batch, row, NH, NH, HS); // qNH = kvNH
}

/*
 * MQA attention
 * @param out: output matrix(batch, row, col) where col = qNH * HS
 * @param inp: input matrix(batch, row, (qNH + 2 * kvNH) * HS) (Q, K, V) concatenated along the last dimension
 * @param batch: batch size
 * @param row: row size
 * @param qNH: number of Q heads
 * @param HS: head size
 */
void cuda_mqa_attention(void *out, const void *inp, int batch, int row, int qNH, int HS)
{
    return cuda_gqa_attention(out, inp, batch, row, qNH, 1, HS); // kvNH = 1
}

/*
 * GQA attention
 * @param out: output matrix(batch, row, col) where col = qNH * HS
 * @param inp: input matrix(batch, row, (qNH + 2 * kvNH) * HS) (Q, K, V) concatenated along the last dimension
 * @param batch: batch size
 * @param row: row size
 * @param qNH: number of Q heads
 * @param kvNH: number of K and V heads
 * @param HS: head size
 */
void cuda_gqa_attention(void *out, const void *inp, int batch, int row, int qNH, int kvNH, int HS)
{
    float *workspace, *qkv, *att;
    int gNH = qNH / kvNH;  // number of grouped query heads

    size_t workspace_size = batch * qNH * row * HS * sizeof(float);
    size_t qkv_size = (batch * row * (qNH + 2 * kvNH) * HS) * sizeof(float);
    size_t att_size = batch * gNH * kvNH * row * row * sizeof(float);

    cuda_check(cudaMalloc(&workspace, workspace_size));
    cuda_check(cudaMalloc(&qkv, qkv_size));
    cuda_check(cudaMalloc(&att, att_size));

    float *q = qkv;
    float *k = qkv + batch * row * qNH * HS;
    float *v = k + batch * row * kvNH * HS;

    // 1. Permute and separate input
    // q: (batch, row, qNH, HS) -> (batch, gNH, kvNH, row, HS)
    // k: (batch, row, kvNH, HS) -> (batch, kvNH, row, HS)
    // v: (batch, row, kvNH, HS) -> (batch, kvNH, row, HS)
    int total_threads = batch * qNH * row * HS;
    int block_size = 256;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    gqa_permute_kernel<<<num_blocks, block_size>>>(q, k, v, (const float*)inp,
                                                  batch, row, qNH, kvNH, HS);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 2. Batched matrix multiplication: Q @ K^T  # (batch, gNH, kvNH, row, HS) @ (batch, kvNH, HS, row) -> (batch, gNH, kvNH, row, row)
    for (int g = 0; g < gNH; g++) {
        float* q_group = q + g * kvNH * batch * row * HS;
        cublas_check(cublasSgemmStridedBatched(cublas_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                row, row, HS,
                                &alpha,
                                k, HS, row * HS,
                                q_group, HS, row * HS,
                                &beta,
                                att + g * kvNH * batch * row * row,
                                row, row * row,
                                batch * kvNH));
    }

    // 3. Apply scaled softmax with causal masking
    float scale = 1.0f / sqrtf(HS);
    int softmax_block_size = 256;
    size_t shared_mem_size = 2 * (softmax_block_size / 32) * sizeof(float);
    int grid_size = batch * gNH * kvNH * row;
    scaled_softmax_kernel<<<grid_size, softmax_block_size, shared_mem_size>>>(
        att, att, batch * gNH, kvNH, row, scale);

    // 4. Batched matrix multiplication: attention @ V # (batch, gNH, kvNH, row, row) @ (batch, kvNH, row, HS) -> (batch, gNH, kvNH, row, HS)
    for (int g = 0; g < gNH; g++) {
        float* workspace_group = workspace + g * kvNH * batch * row * HS;
        cublas_check(cublasSgemmStridedBatched(cublas_handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                HS, row, row,
                                &alpha,
                                v, HS, row * HS,
                                att + g * kvNH * batch * row * row,
                                row, row * row,
                                &beta,
                                workspace_group, HS, row * HS,
                                batch * kvNH));
    }
    // 5. Unpermute result # (batch, gNH, kvNH, row, HS) -> (batch, row, qNH, HS)
    num_blocks = CEIL_DIV(batch * row * qNH * HS, block_size);
    gqa_unpermute_kernel<<<num_blocks, block_size>>>(workspace, (float *)out, batch, row, qNH, kvNH, HS);

    cuda_check(cudaFree(workspace));
    cuda_check(cudaFree(qkv));
    cuda_check(cudaFree(att));
}

}