#include "wukong.h"

#define QK_K 256

dtype_info dtype_infos[GGML_TYPE_COUNT] = {
    {"F32", 1, 4},
    {"F16", 1, 2},
    {"GGML_TYPE_Q4_0", 32, 2 + 16},
    {"GGML_TYPE_Q4_1", 32, 2 + 2 + 16},
    {"GGML_TYPE_Q4_2", 32, 2 + 2 + 16},
    {"GGML_TYPE_Q4_3", 32, 2 + 2 + 16},
    {"GGML_TYPE_Q5_0", 32, 2 + 4 + 16},
    {"GGML_TYPE_Q5_1", 32, 2 + 2 + 4 + 16},
    {"GGML_TYPE_Q8_0", 32, 2 + 32},
    {"GGML_TYPE_Q8_1", 32, 4 + 4 + 32},
    {"GGML_TYPE_Q2_K", 256, 2 + 2 + QK_K / 16 + QK_K / 4},
    {"GGML_TYPE_Q3_K", 256, 2 + QK_K / 4 + QK_K / 8 + 12},
    {"GGML_TYPE_Q4_K", 256, 2 + 2 + QK_K / 2 + 12},
    {"GGML_TYPE_Q5_K", 256, 2 + 2 + QK_K / 2 + QK_K / 8 + 12},
    {"GGML_TYPE_Q6_K", 256, 2 + QK_K / 2 + QK_K / 4 + QK_K / 16},
    {"GGML_TYPE_Q8_K", 256, 4 + QK_K + QK_K / 8},
    {"GGML_TYPE_IQ2_XXS", 256, 2 + QK_K / 4},
    {"GGML_TYPE_IQ2_XS", 256, 2 + QK_K / 4 + QK_K / 32},
    {"GGML_TYPE_IQ3_XXS", 256, 2 + QK_K / 4 + QK_K / 8},
    {"GGML_TYPE_IQ1_S", 256, 2 + QK_K / 8 + QK_K / 16},
    {"GGML_TYPE_IQ4_NL", 32, 2 + 16},
    {"GGML_TYPE_IQ3_S", 256, 2 + QK_K / 4 + QK_K / 8 + QK_K / 32 + 4},
    {"GGML_TYPE_IQ2_S", 256, 2 + QK_K / 4 + QK_K / 16},
    {"GGML_TYPE_IQ4_XS", 256, 2 + 2 + QK_K / 2 + QK_K / 64},
    {"Int8", 1, 1},
    {"Int16", 1, 2},
    {"Int32", 1, 4},
    {"Int64", 1, 8},
    {"F64", 1, 8},
    {"GGML_TYPE_IQ1_M", 256, QK_K / 8 + QK_K / 16 + QK_K / 32},
    {"BF16", 1, 2},
    {"GGML_TYPE_Q4_0_4_4", 32, 2 + 16},
    {"GGML_TYPE_Q4_0_4_8", 32, 2 + 16},
    {"GGML_TYPE_Q4_0_8_8", 32, 2 + 16},
    {"GGML_TYPE_TQ1_0", 256, 2 + 4 * 13},
    {"GGML_TYPE_TQ2_0", 256, 2 + 64}
};

// cuBLAS workspace. Only Hopper needs 32 MB, for others 4 is OK
static size_t cublaslt_workspace_size = 4 * 1024 * 1024;
static void* cublaslt_workspace = NULL;
static cublasComputeType_t cublas_compute_type;
static cublasLtHandle_t cublaslt_handle;
static cublasHandle_t cublas_handle;
static cudaStream_t main_stream;
static int deviceIdx = 0;
__attribute_maybe_unused__ static int cuda_arch_major = 0;
__attribute_maybe_unused__ static int cuda_arch_minor = 0;
__attribute_maybe_unused__ static int cuda_num_SMs = 0; // for persistent threads where we want 1 threadblock per SM
__attribute_maybe_unused__ static int cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU
__attribute_maybe_unused__ static int cuda_threads_per_block = 0;
__attribute_maybe_unused__ static int cuda_warp_size = 0; // warp size of the GPU
__attribute_maybe_unused__ static int cuda_max_shared_mem_per_block = 0;

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ void warp_reduce_max(float& val, int& idx) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

__global__ void softmax_kernel(float* output, const float* input, int row, int col) {
    extern __shared__ float shared_mem[];
    float* row_max = shared_mem;                    // First part of shared memory for max values
    float* row_sum = &shared_mem[blockDim.x / WARP_SIZE];  // Second part for sum values

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    int row_idx = blockIdx.x;

    if (row_idx >= row) return;

    // Step 1: Find maximum value in the row
    float thread_max = -INFINITY;
    for (int i = tid; i < col; i += blockDim.x) {
        thread_max = fmaxf(thread_max, input[row_idx * col + i]);
    }

    // Warp-level reduction for max
    thread_max = warp_reduce_max(thread_max);

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

__global__ void add_bias_kernel(float* out, const float* bias, int T, int OC)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < T * OC; i += stride) {
        int col = i % OC;
        out[i] += bias[col];
    }
}

template <typename T>
__device__ __forceinline__ T sigmoid(const T x)
{
    return 1.0f / (1.0f + expf(type_to_float(-x)));
}

template <typename T>
__global__ void swiglu_kernel(T* out, const T* inp, int B, int TT, int C)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * Packed128<T>::size;
    int total_elements = B * TT * C;

    if (idx >= total_elements) return;

    // Calculate batch, sequence position and channel indices
    int b = idx / (TT * C);
    int t = (idx / C) % TT;
    int c_base = idx % C;

    // Calculate base indices for fc1 and fc2 parts
    int fc_idx_base = (b * TT * 2 * C) + (t * 2 * C) + c_base;

    Packed128<T> packed_fc1 = load128cs(inp + fc_idx_base);
    Packed128<T> packed_fc2 = load128cs(inp + fc_idx_base + C);
    Packed128<T> packed_out;

    #pragma unroll
    for (int i = 0; i < Packed128<T>::size; i++) {
        float val1 = type_to_float(packed_fc1[i]);
        float val2 = type_to_float(packed_fc2[i]);
        float swish_val = val2 * sigmoid(val2);
        packed_out[i] = float_to_type<T>(swish_val * val1);
    }

    store128(out + idx, packed_out);
}

template <typename T>
__global__ void rope_qkv_kernel(T *out, const T *inp, const float *freqs,
                               int batch, int row, int NH, int kvNH, int HS)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HS_half = HS / 2;
    int total_heads = NH + kvNH;
    int total = batch * row * total_heads * HS_half;

    if (idx >= total)
        return;

    int b = idx / (row * total_heads * HS_half);
    int r = (idx / (total_heads * HS_half)) % row;
    int h = (idx / HS_half) % total_heads;
    int d = idx % HS_half;

    int freq_idx = r * HS + 2 * d;
    float c = freqs[freq_idx];
    float s = freqs[freq_idx + 1];

    int base = b * (row * (NH + 2 * kvNH) * HS) + r * ((NH + 2 * kvNH) * HS) + h * HS + 2 * d;

    float x_read = type_to_float<T>(inp[base]);
    float x_imag = type_to_float<T>(inp[base + 1]);
    float result_real = x_read * c - x_imag * s;
    float result_imag = x_read * s + x_imag * c;
    out[base] = float_to_type<T>(result_real);
    out[base + 1] = float_to_type<T>(result_imag);
}

__global__ void get_embeddings_kernel(void* out, const int* inp, const void* embd, int batch, int row, size_t bytes_per_row)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * row)
        return;

    int b = idx / row;
    int t = idx % row;
    int ix = inp[b * row + t];

    char* dst = (char*)out + (b * row + t) * bytes_per_row;
    const char* src = (const char*)embd + ix * bytes_per_row;

    memcpy(dst, src, bytes_per_row);
}

void cuda_matmul_cublas(float *out, const float *inp, const float *weight, const float *bias,
                        int row, int column, int oc)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // cublas sees us transposed, so we want out(oc, row) = weight(oc, c) @ inp(c, row) + bias
    cublas_check(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, oc, row, column, /* M, N, K*/
                            &alpha, weight, oc, inp, column, &beta, out, oc));
    if (bias != NULL) {
        int block_size = cuda_threads_per_block;
        int grid_size = CEIL_DIV(oc * row, block_size);
        add_bias_kernel<<<grid_size, block_size, 0, main_stream>>>(out, bias, row, oc);
        cuda_check(cudaGetLastError());
    }
}

template<typename T>
void cuda_matmul_cublaslt(T *out, const T *inp, const T *weight, const T *bias,
                        int row, int column, int oc)
{
    int res;
    bool has_bias = (bias != nullptr);
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulPreference_t pref;
    cublasLtMatrixLayout_t inp_layout, weight_layout, out_layout, bias_layout;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasOperation_t notrans = CUBLAS_OP_N;
    cublasOperation_t trans = CUBLAS_OP_T;
    cublasLtEpilogue_t epilogue = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
    cudaDataType_t data_type;
    cublasComputeType_t compute_type;

    if constexpr (std::is_same<T, float>::value) {
        data_type = CUDA_R_32F;
        compute_type = cublas_compute_type;
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
        data_type = CUDA_R_16BF;
        compute_type = CUBLAS_COMPUTE_32F;
        // Forces any reductions during matrix multiplications to use the compute type and not the output type
        cublasSetMathMode(cublas_handle, (cublasMath_t)(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    } else if constexpr (std::is_same<T, half>::value) {
        data_type = CUDA_R_16F;
        compute_type = CUBLAS_COMPUTE_32F;
    } else {
        panic("Unsupported type for cuda_matmul_cublaslt");
    }

    /*
     * Cuda is colum-major, for row-major Array, if we want to get: out = inp @ weight.T, 'out' should be 'out.T'.
     * Mathematically, out.T = weight @ inp.T. Since cuda is colum-major, 'weight' should be weight.T, 'inp.T' should be inp.
     * so calculating out.T = weight.T & inp.
     */
    cublas_check(cublasLtMatmulDescCreate(&desc, compute_type, CUDA_R_32F));
    cublas_check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    cublas_check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &notrans, sizeof(notrans)));
    cublas_check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    cublas_check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    cublas_check(cublasLtMatrixLayoutCreate(&weight_layout, data_type, column, oc, column));
    cublas_check(cublasLtMatrixLayoutCreate(&inp_layout, data_type, column, row, column));
    cublas_check(cublasLtMatrixLayoutCreate(&out_layout, data_type, oc, row, oc));
    cublas_check(cublasLtMatrixLayoutCreate(&bias_layout, data_type, oc, 1, oc));

    if (has_bias && (uintptr_t)bias % 16 != 0)
        panic("bias must be aligned to 16 bytes");

    cublas_check(cublasLtMatmulPreferenceCreate(&pref));
    cublas_check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    cublas_check(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, desc, weight_layout, inp_layout, out_layout,
                out_layout, pref, 1, &heuristic, &res));
    if (res == 0)
        panic("No algorithm found: row=%d, column=%d, oc=%d, has_bias=%d", row, column, oc, has_bias);

    const float alpha = 1.0f, beta = 0.0f;
    cublas_check(cublasLtMatmul(cublaslt_handle, desc, &alpha, weight, weight_layout, inp, inp_layout, &beta,
                out, out_layout, out, out_layout, &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, main_stream));

    cublas_check(cublasLtMatmulPreferenceDestroy(pref));
    cublas_check(cublasLtMatmulDescDestroy(desc));
    cublas_check(cublasLtMatrixLayoutDestroy(weight_layout));
    cublas_check(cublasLtMatrixLayoutDestroy(inp_layout));
    cublas_check(cublasLtMatrixLayoutDestroy(out_layout));
    cublas_check(cublasLtMatrixLayoutDestroy(bias_layout));
}

void cuda_matmul_cublaslt_f32(void *out, const void *inp, const void *weight, const void *bias,
                        int row, int column, int oc)
{
    cuda_matmul_cublaslt<float>(
        static_cast<float*>(out),
        static_cast<const float*>(inp),
        static_cast<const float*>(weight),
        static_cast<const float*>(bias),
        row, column, oc
    );
}

void cuda_matmul_cublaslt_bf16(void *out, const void *inp, const void *weight, const void *bias,
                        int row, int column, int oc)
{
    cuda_matmul_cublaslt<nv_bfloat16>(
        static_cast<nv_bfloat16*>(out),
        static_cast<const nv_bfloat16*>(inp),
        static_cast<const nv_bfloat16*>(weight),
        static_cast<const nv_bfloat16*>(bias),
        row, column, oc
    );
}

void cuda_matmul_cublaslt_f16(void *out, const void *inp, const void *weight, const void *bias,
                        int row, int column, int oc)
{
    cuda_matmul_cublaslt<half>(
        static_cast<half*>(out),
        static_cast<const half*>(inp),
        static_cast<const half*>(weight),
        static_cast<const half*>(bias),
        row, column, oc
    );
}

__global__ void div_kernel(float *out, const float *a, const float *b, int row, int col)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = row * col;

    if (idx < total_elements) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void dequantize_Q8_0(float *out, const block_q8_0 *inp, int row, int nb, int bs)
{
    extern __shared__ block_q8_0 shared_block[];
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = row * nb;

    if (block_idx >= total_blocks)
        return;

    int r = block_idx / nb; // row index
    int b = block_idx % nb; // block index

    const block_q8_0 *block = inp + r * nb + b;
    shared_block[threadIdx.x] = *block;
    __syncthreads();

    float scale = __half2float(shared_block[threadIdx.x].scale);
    #pragma unroll
    for (int i = 0; i < bs; ++i) {
	    int out_idx = r * nb * bs + b * bs + i;
	    out[out_idx] = scale * shared_block[threadIdx.x].d[i];
    }
}

template <typename T>
__global__ void add_kernel(T* out, const T* a, const T* b, int row, int col)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * Packed128<T>::size;
    int size = row * col;

    if (idx >= size)
        return;
    Packed128<T> packed_a = load128cs(a + idx);
    Packed128<T> packed_b = load128cs(b + idx);
    Packed128<T> packed_out = packed_a + packed_b;
    store128(out + idx, packed_out);
}

template <typename T>
__global__ void repeat_qkv_kernel(T* __restrict__ replicated_qkv, const T* __restrict__ gqa_qkv,
                               int B, int N, int NH, int HD, int replicate_factor) {
    // we have a single tensor gqa_qkv of shape (B, N, (NH + 2*(NH/replicate_factor)) * HD)
    // we want to replicate it into (B, N, 3 * NH * HD)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N * 3 * NH * HD) { return; }
    int idx_flat = idx; // keep backup

    // decode the output index
    int d = idx % HD;
    idx /= HD;
    int nh = idx % NH;
    idx /= NH;
    int c = idx % 3;
    idx /= 3;
    int n = idx % N;
    int b = idx / N;

    int inp_idx;
    int nh_total = NH + 2 * (NH / replicate_factor);
    if (c == 0) {
        inp_idx = b * N * nh_total * HD + n * nh_total * HD + 0 * NH * HD + nh * HD + d;
    } else if (c == 1) {
        inp_idx = b * N * nh_total * HD + n * nh_total * HD + 1 * NH * HD + (nh / replicate_factor) * HD + d;
    } else {
        inp_idx = b * N * nh_total * HD + n * nh_total * HD + (NH * HD + (NH / replicate_factor) * HD) + (nh / replicate_factor) * HD + d;
    }

    replicated_qkv[idx_flat] = __ldcs(&gqa_qkv[inp_idx]);
}

template <typename T>
__global__ void get_row_kernel(T *out, const T *inp, int batch, int row, int col, int idx)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch) {
        const T *src = inp + b * row * col + idx * col;
        T *dst = out + b * col;
        memcpy(dst, src, col * sizeof(T));
    }
}

template <typename T>
__global__ void argmax_kernel(int *out, const T *inp, int row, int col)
{
    __shared__ float smax[WARP_SIZE];  // Max values per warp
    __shared__ int sidx[WARP_SIZE];    // Corresponding indices

    int r = blockIdx.x;                // One row per block
    int tid = threadIdx.x;             // Thread ID
    int wid = tid / WARP_SIZE;         // Warp ID
    int lane = tid % WARP_SIZE;        // Lane within warp

    if (r >= row) return;

    // Each thread's running max
    float max_val = -INFINITY;
    int max_idx = -1;

    int size = Packed128<T>::size;
    for (int i = tid; i < col / size; i += blockDim.x) {
        Packed128<T> packed_vals = load128cs(&inp[r * col + i * size]);
        #pragma unroll
        for (int j = 0; j < size; j++) {
            float val = type_to_float<T>(packed_vals[j]);
            if (val > max_val) {
                max_val = val;
                max_idx = i * size + j;
            }
        }
    }

    // Warp-level reduction
    warp_reduce_max(max_val, max_idx);

    // Write warp results to shared memory
    if (lane == 0) {
        smax[wid] = max_val;
        sidx[wid] = max_idx;
    }
    __syncthreads();

    // Final reduction across warps by first warp
    if (wid == 0) {
        max_val = (lane < blockDim.x/WARP_SIZE) ? smax[lane] : -INFINITY;
        max_idx = (lane < blockDim.x/WARP_SIZE) ? sidx[lane] : -1;

        warp_reduce_max(max_val, max_idx);

        if (lane == 0) {
            out[r] = max_idx;
        }
    }
}

template <typename T>
__global__ void rmsnorm_kernel(T* __restrict__ out, const T* __restrict__ inp,
                              const float* __restrict__ weight, int N, int C, float eps)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);

    __shared__ float shared_sum2[WARP_SIZE]; // One element per warp for squared sum

    int num_warps = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int idx = blockIdx.x; // One block per row

    // Point to current sequence position
    const T* x = inp + idx * C;

    // Thread coarsening through the row
    float thread_sum2 = 0.0f;

    // Calculate the number of elements to process per thread using Packed128 (4 elements at a time)
    int packed_size = Packed128<T>::size;

    // Each thread accumulates multiple elements using Packed128
    for (int i = threadIdx.x; i < C / packed_size; i += blockDim.x) {
        Packed128<T> packed_xi = load128cs(x + i * packed_size);

        #pragma unroll
        for (int j = 0; j < packed_size; j++) {
            float val = type_to_float<T>(packed_xi[j]);
            thread_sum2 += val * val;
        }
    }

    // Warp-level reduction for sum of squares
    float warp_sum2 = cg::reduce(warp, thread_sum2, cg::plus<float>{});

    // Store warp-level results to shared memory
    if (lane_id == 0) {
        shared_sum2[warp_id] = warp_sum2;
    }
    __syncthreads();

    // Load results from shared memory to threads, pad with zeros for out-of-bounds threads
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;

    // Reduce the warp-level results
    float block_sum2 = cg::reduce(warp, warp_sum2, cg::plus<float>{});

    block_sum2 /= C; // mean(x**2)
    float s = rsqrtf(block_sum2 + eps); // 1 / sqrt(mean(x**2) + eps)

    // Apply normalization and scaling using Packed128 for better memory access
    T* o = out + idx * C;

    for (int i = threadIdx.x; i < C / packed_size; i += blockDim.x) {
        Packed128<T> packed_val = load128cs(x + i * packed_size);
        Packed128<T> packed_result;

        #pragma unroll
        for (int j = 0; j < packed_size; j++) {
            float val = type_to_float<T>(packed_val[j]);
            float normalized = val * s * weight[i * packed_size + j]; // x / sqrt(mean(x**2) + eps) * weight
            packed_result[j] = float_to_type<T>(normalized);
        }
        store128(o + i * packed_size, packed_result);
    }
}

template <typename T>
void cuda_rmsnorm(T *out, const T *inp, const float *weight, int row, int col, float eps)
{
    const int block_size = 256;

    int size = Packed128<T>::size;
    assert(col % size == 0 && "Column size must be a multiple of Packed128<T>::size for cuda_rmsnorm");
    rmsnorm_kernel<T><<<row, block_size, 0, main_stream>>>(out, inp, weight, row, col, eps);
    cuda_check(cudaGetLastError());
}

template<typename T>
void cuda_swiglu(T *out, const T *inp, int batch, int row, int col)
{
    int block_size = 256;
    int size = Packed128<T>::size;

    assert(col % size == 0 && "Column size must be a multiple of Packed128<T>::size for cuda_swiglu");
    // Each thread now processes Packed128<T>::size elements
    int grid_size = CEIL_DIV(batch * row * col / size, block_size);

    swiglu_kernel<T><<<grid_size, block_size, 0, main_stream>>>((T *)out, (const T *)inp, batch, row, col);
    cuda_check(cudaGetLastError());
}

template<typename T>
__global__ void permute_kernel(T* q, T* k, T* v,
                               const T* inp,
                               int B, int N, int NH, int d)
{
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) {
        return;
    }

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    q[idx] = __ldcs(&inp[inp_idx]);
    k[idx] = __ldcs(&inp[inp_idx + NH * d]);
    v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);
}

template<typename T>
__global__ void unpermute_kernel(T* __restrict__ out, const T* __restrict__ inp, int B, int N, int NH, int d)
{
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

// Handles both scaling of attention scores and softmax computation with causal masking
// inp/out shape: (B, NH, T, T)
template<typename T>
__global__ void scaled_softmax_kernel(T* out, const T* inp, int B, int NH, int TT, float scale)
 {
    extern __shared__ float shared[];
    int batch_idx = blockIdx.x / (NH * TT); // batch index
    int head_idx = (blockIdx.x / TT) % NH;  // head index
    int row_idx = blockIdx.x % TT;          // row index within the attention matrix
    int tid = threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;         // warp index within a block
    int laneId = threadIdx.x % WARP_SIZE;         // thread index within a warp
    int warpsPerBlock = blockDim.x / WARP_SIZE;

    // shared memory layout: first half for max values, second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // calculate base index for this thread block's row
    int row_start = (batch_idx * NH * TT * TT) + (head_idx * TT * TT) + (row_idx * TT);
    const T* x = inp + row_start;

    // Step 1: Find maximum while applying scale and causal mask
    float maxval = -INFINITY;
    for (int i = tid; i < TT; i += blockDim.x) {
        float val = (i <= row_idx) ? type_to_float<T>(x[i]) * scale : -INFINITY;
        maxval = fmaxf(maxval, val);
    }

    // warp-level reduction for maxval
    maxval = warp_reduce_max(maxval);

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
    for (int i = tid; i < TT; i += blockDim.x) {
        float val;
        if (i <= row_idx) {
            val = expf(type_to_float<T>(x[i]) * scale - offset);
        } else {
            val = 0.0f;
        }

        out[row_start + i] = float_to_type<T>(val);  // store intermediate result
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
    float inv_sum = 1.0f / sumvals[0];

    // write final normalized values
    for (int i = tid; i < TT; i += blockDim.x) {
        if (i <= row_idx) {
            float val = type_to_float<T>(out[row_start + i]) * inv_sum;
            out[row_start + i] = float_to_type<T>(val);
        } else {
            out[row_start + i] = float_to_type<T>(0.0f);
        }
    }
}

template<typename T>
void cuda_mh_sdpa(T *out, const T *inp, int batch, int row, int NH, int HS)
{
    T *qkv, *att, *vatt;

    // Allocate space for broadcasted K and V
    size_t q_size = (batch * NH * row * HS) * sizeof(T);
    size_t qkv_size = 3 * q_size;
    size_t att_size = batch * NH * row * row * sizeof(T);

    qkv = (T *)cuda_malloc(qkv_size);
    // try best to reuse input buffer
    vatt = (T *)inp;
    att = (T *)cuda_malloc(att_size);

    T *q = qkv;
    T *k = qkv + batch * NH * row * HS;
    T *v = k + batch * NH * row * HS;

    // Permute input
    // q: (batch, row, NH, HS) -> (batch, NH, row, HS)
    // k: (batch, row, NH, HS) -> (batch, NH, row, HS)
    // v: (batch, row, NH, HS) -> (batch, NH, row, HS)
    int total_threads = batch * NH * row * HS;
    int block_size = 256;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<T><<<num_blocks, block_size, 0, main_stream>>>(q, k, v, inp, batch, row, NH, HS);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasComputeType_t compute_type = cublas_compute_type;

    cudaDataType data_type;
    if constexpr (std::is_same<T, float>::value) {
        data_type = CUDA_R_32F;
    } else if constexpr (std::is_same<T, nv_bfloat16>::value) {
        data_type = CUDA_R_16BF;
        compute_type = CUBLAS_COMPUTE_32F;
    } else if constexpr (std::is_same<T, half>::value) {
        data_type = CUDA_R_16F;
        compute_type = CUBLAS_COMPUTE_32F;
    } else {
        panic("Unsupported type for cuda_mh_sdpa");
    }

    // Batched matrix multiplication: Q @ K^T
    cublas_check(cublasGemmStridedBatchedEx(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            row, row, HS,
                            &alpha,
                            k, data_type, HS, row * HS,
                            q, data_type, HS, row * HS,
                            &beta,
                            att, data_type, row, row * row,
                            batch * NH,
                            compute_type, CUBLAS_GEMM_DEFAULT
                        ));

    // Apply scaled softmax with causal masking
    float scale = 1.0f / sqrtf(HS);
    int softmax_block_size = 256;
    size_t shared_mem_size = 2 * (softmax_block_size / 32) * sizeof(float);
    int grid_size = batch * NH * row;
    scaled_softmax_kernel<T><<<grid_size, softmax_block_size, shared_mem_size, main_stream>>>(
        att, att, batch, NH, row, scale);

    // Batched matrix multiplication: attention @ V
    cublas_check(cublasGemmStridedBatchedEx(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, row, row,
                            &alpha,
                            v, data_type, HS, row * HS,
                            att, data_type, row, row * row,
                            &beta,
                            vatt, data_type, HS, row * HS,
                            batch * NH,
                            compute_type, CUBLAS_GEMM_DEFAULT
                        ));

    // Unpermute result from (batch, NH, row, HS) -> (batch, row, NH, HS)
    num_blocks = CEIL_DIV(batch * row * NH * HS, block_size);
    unpermute_kernel<T><<<num_blocks, block_size, 0, main_stream>>>(out, vatt, batch, row, NH, HS);

    cuda_free(qkv);
    cuda_free(att);
}

template <typename T>
void cuda_rope_qkv(T *out, const T *inp, const float *freqs, int batch, int row, int NH, int kvNH, int HS)
{
    int block_size = 256;
    // We only need threads for Q and K sections, V will be untouched
    int total_threads = batch * row * (NH + kvNH) * HS / 2;
    int num_blocks = CEIL_DIV(total_threads, block_size);

    rope_qkv_kernel<T><<<num_blocks, block_size, 0, main_stream>>>(out, inp, freqs, batch, row, NH, kvNH, HS);
    cuda_check(cudaGetLastError());
}

template <typename T>
void cuda_add(T *out, const T *a, const T *b, int row, int col)
{
    const int total_size = row * col;
    const int block_size = 256;

    int size = Packed128<T>::size;

    if (col % size != 0) {
        panic("Column size must be a multiple of %d for cuda_add", size);
    }

    // Each thread handles 4 elements when using vectorized operations
    const int grid_size = CEIL_DIV(total_size / size, block_size);

    add_kernel<T><<<grid_size, block_size, 0, main_stream>>>(
        out,
        a,
        b,
        row, col);

    cuda_check(cudaGetLastError());
}

template <typename T>
void cuda_repeat_qkv(T *out, const T *inp, int batch, int row, int qNH, int kvNH, int HS)
{
    const int block_size = 256;
    int total_threads = batch * row * (3 * qNH) * HS; // one thread per output element
    int num_blocks = CEIL_DIV(total_threads, block_size);
    int replicate_factor = qNH / kvNH;
    assert(replicate_factor > 1);
    repeat_qkv_kernel<T><<<num_blocks, block_size, 0, main_stream>>>(out, inp, batch, row, qNH, HS, replicate_factor);
    cuda_check(cudaGetLastError());
}

template <typename T>
void group_query_attention(T *out, const T *embeds, const void *freqs, const void *norm_weight, const T *qkv_weight,
                            const T *out_weight, int batch, int row, int NH, int kvNH, int HS, float eps, int dtype)
{
    T *qkv, *att, *output, *r_qkv;
    int col = NH * HS;
    int qkv_weight_row = (NH + 2 * kvNH) * HS;
    att = (T *)cuda_malloc(batch * row * col * sizeof(T));
    output = (T *)cuda_malloc(batch * row * col * sizeof(T));
    qkv = (T *)cuda_malloc(batch * row * qkv_weight_row * sizeof(T));
    r_qkv = (T *)cuda_malloc(batch * row * 3 * NH * HS * sizeof(T));

    cuda_rmsnorm<T>(att, embeds, (const float *)norm_weight, batch * row, col, eps);
    cuda_matmul(qkv, att, qkv_weight, nullptr, batch * row, col, qkv_weight_row, dtype); // (batch * row, col) @ (qkv_weight_row, col)^T
    cuda_rope_qkv<T>(qkv, qkv, (const float *)freqs, batch, row, NH, kvNH, HS); // rope qkv in-place
    cuda_repeat_qkv<T>(r_qkv, qkv, batch, row, NH, kvNH, HS);
    cuda_mh_sdpa<T>(att, r_qkv, batch, row, NH, HS);
    cuda_matmul(output, att, out_weight, nullptr, batch * row, col, col, dtype); // (batch * row, col) @ (col, col)^T
    cuda_add<T>(out, embeds, output, batch * row, col); // residual connect embeddings to attention

    cuda_free(qkv);
    cuda_free(att);
    cuda_free(output);
    cuda_free(r_qkv);
}

template<typename T>
void feed_forward(T *out, const T *attn, const float *norm_weight, const T *fc_weight, const T *out_weight,
                    int batch, int row, int col, int ffl, float eps, int dtype)
{
    T *ffn, *fc;

    ffn = (T *)cuda_malloc(batch * row * col * sizeof(T));
    fc = (T *)cuda_malloc(batch * row * 2 * ffl * sizeof(T));

    cuda_rmsnorm<T>(ffn, attn, norm_weight, batch * row, col, eps);
    cuda_matmul(fc, ffn, fc_weight, nullptr, batch * row, col, 2 * ffl, dtype); // (batch * row, col) @ (2 * ffl, col)^T
    cuda_swiglu<T>(fc, fc, batch, row, ffl); // update fc in-place
    cuda_matmul(ffn, fc, out_weight, nullptr, batch * row, ffl, col, dtype); // (batch * row, ffl) @ (col, ffl)^T
    cuda_add<T>(out, attn, ffn, batch * row, col); // residual connect attention to feedforward

    cuda_free(fc);
    cuda_free(ffn);
}

template <typename T>
void cuda_get_row(T *out, const T *inp, int batch, int row, int col, int idx)
{
    const int block_size = 8;
    const int total_threads = batch;
    const int grid_size = CEIL_DIV(total_threads, block_size);

    if (idx < 0)
        idx += row;
    assert(idx >= 0 && idx < row);
    get_row_kernel<T><<<grid_size, block_size, 0, main_stream>>>(out, inp, batch, row, col, idx);
    cuda_check(cudaGetLastError());
}

template <typename T>
void cuda_argmax(int *out, const T *inp, int row, int col)
{
    const int block_size = 256;
    const int grid_size = row;

    // Ensure column size is a multiple of Packed128<T>::size
    int size = Packed128<T>::size;
    assert(col % size == 0 && "Column size must be a multiple of Packed128<T>::size for cuda_argmax");

    argmax_kernel<T><<<grid_size, block_size, 0, main_stream>>>(out, inp, row, col);
    cuda_check(cudaGetLastError());
}

template <typename T>
void classify(T *out, T *ff, const float *norm_weight, const T *out_weight, int batch, int row, int col, int wsize, float eps, int dtype)
{
    T *ffn = (T *)ff + batch * col; // reuse the memory of ff

    assert(batch * 2 <= row);
    cuda_get_row<T>(ff, ff, batch, row, col, -1); // out shape: (batch, col)
    cuda_rmsnorm<T>(ffn, ff, (const float *)norm_weight, batch, col, eps);
    cuda_matmul(out, ffn, out_weight, nullptr, batch, col, wsize, dtype); // (batch, col) @ (wsize, col)^T
}

template <typename T>
void predict(T *out, T *ff, const float *norm_weight, const T *out_weight, int batch, int row, int col, int wsize, float eps, int dtype)
{
    T *logits = (T *)cuda_malloc(batch * wsize * sizeof(T));
    classify<T>(logits, ff, norm_weight, out_weight, batch, row, col, wsize, eps, dtype);
    cuda_argmax<T>((int *)out, logits, batch, wsize); // TODO: support temp, top_k and top_p
    cuda_free(logits);
}

extern "C" {
void cuda_init(int idx)
{
    srand(0);   // determinism

    deviceIdx = idx;

    cuda_check(cudaSetDevice(deviceIdx));
    cuda_check(cudaStreamCreate(&main_stream));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_num_SMs = deviceProp.multiProcessorCount;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;
    cuda_threads_per_block = deviceProp.maxThreadsPerBlock;
    cuda_warp_size = deviceProp.warpSize;
    cuda_max_shared_mem_per_block = deviceProp.sharedMemPerBlock;
    // printf("CUDA device: %s, major %d, minor %d, num_SMs: %d, threads_per_SM: %d, threads_per_block: %d, warp_size: %d\n",
    //        deviceProp.name, cuda_arch_major, cuda_arch_minor, cuda_num_SMs, cuda_threads_per_SM, cuda_threads_per_block, cuda_warp_size);
    printf("Running on GPU %d: %s, Compute Capability %d.%d\n", deviceIdx, deviceProp.name, cuda_arch_major, cuda_arch_minor);

    cublas_check(cublasCreate(&cublas_handle));
    cublas_check(cublasLtCreate(&cublaslt_handle));
    cublaslt_workspace = cuda_malloc(cublaslt_workspace_size);

    // Set the stream for cublas handle
    cublas_check(cublasSetStream(cublas_handle, main_stream));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}

void cuda_fini(void)
{
    cuda_free(cublaslt_workspace);
    cublas_check(cublasDestroy(cublas_handle));
    cublas_check(cublasLtDestroy(cublaslt_handle));
    cuda_check(cudaStreamDestroy(main_stream));
}


void* cuda_malloc(size_t size)
{
    void *ptr;
    cuda_check(cudaMallocAsync(&ptr, size, main_stream));
    return ptr;
}

void cuda_free(void* ptr)
{
    cuda_check(cudaFreeAsync(ptr, main_stream));
}

void cuda_to_device(void* dst, void* src, size_t size)
{
    cuda_check(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, main_stream));
}

void cuda_to_host(void* dst, void* src, size_t size)
{

    cuda_check(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, main_stream));
    cuda_check(cudaStreamSynchronize(main_stream));
}

/*
 * Fused matrix multiplication with optional bias addition: out = inp @ weight^T + bias
 *
 * @param out: output matrix(row, oc)
 * @param inp: input matrix(row, column)
 * @param weight: weight matrix(oc, column)
 * @param bias: optional bias vector(oc) (can be NULL)
 * @param row: input row size
 * @param column: input column size
 * @param oc: output column size
 */
void cuda_matmul(void *out, const void *inp, const void *weight, const void *bias,
                int row, int column, int oc, int dtype)
{
    if (dtype == GGML_TYPE_BF16) {
        return cuda_matmul_cublaslt_bf16(out, inp, weight, bias, row, column, oc);
    } else if (dtype == GGML_TYPE_F16) {
        return cuda_matmul_cublaslt_f16(out, inp, weight, bias, row, column, oc);
    } else if (dtype == GGML_TYPE_F32) {
        return cuda_matmul_cublaslt_f32(out, inp, weight, bias, row, column, oc);
    } else {
        void *dw = cuda_malloc(oc * column * sizeof(float));
        cuda_dequantize(dw, weight, oc, column, dtype);
        cuda_matmul_cublaslt_f32(out, inp, dw, bias, row, column, oc);
        cuda_free(dw);
        return;
    }
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
    const int shared_mem_size = (2 * (block_size / WARP_SIZE)) * sizeof(float); // Space for max and sum values
    softmax_kernel<<<row, block_size, shared_mem_size, main_stream>>>((float *)output, (const float *)input, row, col);
    cuda_check(cudaGetLastError());
}

/*
 * GQA scaled dot product attention
 *
 * @param out: output matrix(batch, row, col) where col = qNH * HS
 * @param inp: input matrix(batch, row, (qNH + 2 * kvNH) * HS) (Q, K, V) concatenated along the last dimension
 * @param batch: batch size
 * @param row: row size
 * @param qNH: number of Q heads
 * @param kvNH: number of K and V heads
 * @param HS: head size
 */
void cuda_gq_sdpa(void *out, const void *inp, int batch, int row, int qNH, int kvNH, int HS)
{
    void *qkv = cuda_malloc(batch * row * 3 * qNH * HS * sizeof(float));
    cuda_repeat_qkv(qkv, inp, batch, row, qNH, kvNH, HS);
    cuda_mh_sdpa(out, qkv, batch, row, qNH, HS);
    cuda_free(qkv);
}

/* Root Mean Square Layer Normalization: x / sqrt(mean(x^2) + eps) * weight
 *
 * @param out: output matrix(row, col)
 * @param inp: input matrix(row, col)
 * @param weight: weight matrix(col)
 * @param row: row size
 * @param col: column size
 * @param eps: epsilon value
 */
void cuda_rmsnorm(void *out, const void *inp, const void *weight, int row, int col, float eps)
{
    cuda_rmsnorm<float>((float *)out, (const float *)inp, (const float *)weight, row, col, eps);
}

// swiglu: y = swish(fc2(x)) * fc1(x), where swish(x) = x / (1 + exp(-x)), fc1 and fc2 are fully connected layers
// @param out: output matrix(batch, row, col)
// @param inp: input matrix(batch, row, 2*col), concatenated fc1 and fc2 outputs along the last dimension
void cuda_swiglu(void *out, const void *inp, int batch, int row, int col)
{
    cuda_swiglu<float>((float *)out, (const float *)inp, batch, row, col);
}

/*
 * Vanilla multi-head scaled dot product attention
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
void cuda_mh_sdpa(void *out, const void *inp, int batch, int row, int NH, int HS)
{
    cuda_mh_sdpa<float>((float*)out, (const float*)inp, batch, row, NH, HS);
}

/*
 * Multi query scaled dot product attention
 *
 * @param out: output matrix(batch, row, col) where col = qNH * HS
 * @param inp: input matrix(batch, row, (qNH + 2 * kvNH) * HS) (Q, K, V) concatenated along the last dimension
 * @param batch: batch size
 * @param row: row size
 * @param qNH: number of Q heads
 * @param HS: head size
 */
void cuda_mq_sdpa(void *out, const void *inp, int batch, int row, int qNH, int HS)
{
    void *qkv = cuda_malloc(batch * row * 3 * qNH * HS * sizeof(float));
    cuda_repeat_qkv(qkv, inp, batch, row, qNH, 1, HS);
    cuda_mh_sdpa(out, qkv, batch, row, qNH, HS);
    cuda_free(qkv);
}

/*
 * RoPE: Rotated Positional Embedding
 *
 * @param out: output matrix(batch, row, NH + 2*kvNH, HS) where NH is for Q and kvNH each for K,V
 * @param inp: input matrix(batch, row, NH + 2*kvNH, HS) q, k, v concatenated along the last dimension
 * @freqs_cis: cos and sin frequencies for each element in q, k
 * @param batch: batch size
 * @param row: row size
 * @param NH: number of query heads
 * @param kvNH: number of key/value heads
 * @param HS: head size
 */
void cuda_rope_qkv(void *out, const void *inp, const void *freqs, int batch, int row, int NH, int kvNH, int HS)
{
    cuda_rope_qkv<float>((float *)out, (const float *)inp, (const float *)freqs, batch, row, NH, kvNH, HS);
}

/*
 * Get the embeddings for the given indices using the embedding table
 *
 * @param out: output matrix(batch, row, col)
 * @param inp: input matrix(batch, row)
 * @param embd: embedding table (vacob_size, col)
 * @param batch: batch size
 * @param row: row size (number of indices)
 * @param col: column size (embedding size)
 */
void cuda_embedding(void* out, const void *inp, const void *embd, int batch, int row, int col, int dtype)
{
    if (dtype < 0 || dtype >= GGML_TYPE_COUNT)
        panic("Unsupported quantization type: %d", dtype);

    auto info = dtype_infos[dtype];
    assert(col % info.block_size == 0);
    size_t bytes_per_row = col / info.block_size * info.type_size;

    const int block_size = 256;
    const int N = batch * row;  // One thread per row
    const int grid_size = CEIL_DIV(N, block_size);

    if (dtype == GGML_TYPE_F32 || dtype == GGML_TYPE_BF16 || dtype == GGML_TYPE_F16) {
        get_embeddings_kernel<<<grid_size, block_size, 0, main_stream>>>(out, (const int*)inp, embd, batch, row, bytes_per_row);
        cuda_check(cudaGetLastError());
        return;
    }
    void *dout = cuda_malloc(batch * row * bytes_per_row);
    get_embeddings_kernel<<<grid_size, block_size, 0, main_stream>>>(dout, (const int*)inp, embd, batch, row, bytes_per_row);
    cuda_dequantize(out, dout, batch * row, col, dtype);
    cuda_check(cudaGetLastError());
    cuda_free(dout);
}

/*
 * Concatenate the input tensors along the first dimension
 *
 * @param out: output matrix(arow + brow, col)
 * @param a: input matrix(arow, col)
 * @param b: input matrix(brow, col)
 * @param arow: row size of a
 * @param brow: row size of b
 * @param col: column size
 */
void cuda_cat(void *out, const void *a, const void *b, int arow, int brow, int col, int dtype)
{
    auto info = dtype_infos[dtype];
    size_t asize = arow * col * info.type_size / info.block_size;
    size_t bsize = brow * col * info.type_size / info.block_size;

    cuda_check(cudaMemcpyAsync(out, a, asize, cudaMemcpyDeviceToDevice, main_stream));
    cuda_check(cudaMemcpyAsync((char *)out + asize, b, bsize, cudaMemcpyDeviceToDevice, main_stream));
}

/*
 * Element-wise division a / b
 *
 * @param out: output matrix(row, col)
 * @param a: input matrix(row, col)
 * @param b: input matrix(row, col)
 * @param row: row size
 * @param col: column size
 */
void cuda_div(void *out, const void *a, const void *b, int row, int col)
{
    int block_size = 256;
    int total_threads = row * col;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    div_kernel<<<num_blocks, block_size, 0, main_stream>>>((float *)out, (const float *)a, (const float *)b, row, col);
    cuda_check(cudaGetLastError());
}

/*
 * Dequantize the quantized input tensor from dtype to float
 *
 * @param out: output matrix(row, col)
 * @param inp: input matrix(row, col)
 * @param row: row size
 * @param col: column size
 * @param type: quantization dtype
 */
void cuda_dequantize(void *out, const void *inp, int row, int col, int type)
{
    if (type < 0 || type >= GGML_TYPE_COUNT)
        panic("Unsupported quantization type: %d", type);

    auto info = dtype_infos[type];
    int nb = col / info.block_size;
    int bs = info.block_size;
    int total_blocks = row * nb;
    int block_size = 256;
    int num_blocks = CEIL_DIV(total_blocks, block_size);
    size_t shared_mem_size = block_size * sizeof(block_q8_0);
    assert(shared_mem_size <= cuda_max_shared_mem_per_block);
    switch (type) {
    case GGML_TYPE_Q8_0:
	    dequantize_Q8_0<<<num_blocks, block_size, shared_mem_size, main_stream>>>((float *)out, (const block_q8_0 *)inp, row, nb, bs);
	    break;
    default:
	    panic("Unsupported quantization type: %s", dtype_infos[type].name);
	}
    cuda_check(cudaGetLastError());
}

/*
 * Element-wise addition out = a + b
 *
 * @param out: output matrix(row, col)
 * @param a: input matrix(row, col)
 * @param b: input matrix(row, col)
 * @param row: row size
 * @param col: column size
 */
void cuda_add(void* out, const void* a, const void* b, int row, int col)
{
    cuda_add<float>((float *)out, (const float *)a, (const float *)b, row, col);
}

void cuda_group_query_attention(void *out, const void *embeds, const void *freqs, const void *norm_weight, const void *qkv_weight,
                                const void *out_weight, int batch, int row, int NH, int kvNH, int HS, float eps, int dtype)
{
    if (dtype == GGML_TYPE_BF16) {
        group_query_attention<nv_bfloat16>((nv_bfloat16 *)out, (const nv_bfloat16 *)embeds, (const float *)freqs,
                                           (const float *)norm_weight, (const nv_bfloat16 *)qkv_weight,
                                           (const nv_bfloat16 *)out_weight, batch, row, NH, kvNH, HS, eps, dtype);
    } else if (dtype == GGML_TYPE_F16) {
        group_query_attention<half>((half *)out, (const half *)embeds, (const float *)freqs, (const float *)norm_weight,
                                    (const half *)qkv_weight, (const half *)out_weight, batch, row, NH, kvNH, HS, eps, dtype);
    } else {
        group_query_attention<float>((float *)out, (const float *)embeds, (const float *)freqs, (const float *)norm_weight,
                                     (const float *)qkv_weight, (const float *)out_weight, batch, row, NH, kvNH, HS, eps, dtype);
    }
}

/*
 * Replicate K, V to match the size of Q
 *
 * @param out: output matrix(batch, row, (3 * qNH) * HS)
 * @param inp: input matrix(batch, row, (qNH + 2 * kvNH) * HS) (Q, K, V) concatenated along the last dimension
 * @param batch: batch size
 * @param row: row size
 * @param qNH: number of Q heads
 * @param kvNH: number of K and V heads
 * @param HS: head size
 */
void cuda_repeat_qkv(void *out, const void *inp, int batch, int row, int qNH, int kvNH, int HS)
{
    cuda_repeat_qkv<float>((float *)out, (const float *)inp, batch, row, qNH, kvNH, HS);
}

/*
 * Get the row at the given index
 *
 * @param out: output matrix(batch, col)
 * @param inp: input matrix(batch, row, col)
 * @param batch: batch size
 * @param row: row index
 * @param col: column size
 * @param idx: index. If negative, it is idx from the end.
 */
void cuda_get_row(void *out, const void *inp, int batch, int row, int col, int idx)
{
    cuda_get_row<float>((float *)out, (const float *)inp, batch, row, col, idx);
}

/*
 * Get the idx of the maximum value along the last dimension
 *
 * @param out: output vector(row) as type int
 * @param inp: input matrix(row, col)
 * @param row: row size
 * @param col: column size
 */
void cuda_argmax(void *out, const void *inp, int row, int col)
{
    cuda_argmax<float>((int *)out, (const float *)inp, row, col);
}

void cuda_feed_forward(void *out, const void *attn, const void *norm_weight, const void *fc_weight, const void *out_weight,
                    int batch, int row, int col, int ffl, float eps, int dtype)
{
    if (dtype == GGML_TYPE_BF16) {
        feed_forward<nv_bfloat16>((nv_bfloat16 *)out, (const nv_bfloat16 *)attn, (const float *)norm_weight,
                                  (const nv_bfloat16 *)fc_weight, (const nv_bfloat16 *)out_weight,
                                  batch, row, col, ffl, eps, dtype);
    } else if (dtype == GGML_TYPE_F16) {
        feed_forward<half>((half *)out, (const half *)attn, (const float *)norm_weight,
                           (const half *)fc_weight, (const half *)out_weight,
                           batch, row, col, ffl, eps, dtype);
    } else {
        feed_forward<float>((float *)out, (const float *)attn, (const float *)norm_weight,
                            (const float *)fc_weight, (const float *)out_weight,
                            batch, row, col, ffl, eps, dtype);
    }
}

void cuda_classify(void *out, void *ff, const void *norm_weight, const void *out_weight, int batch, int row, int col, int wsize, float eps, int dtype)
{
    if (dtype == GGML_TYPE_BF16) {
        classify<nv_bfloat16>((nv_bfloat16 *)out, (nv_bfloat16 *)ff, (const float *)norm_weight, (const nv_bfloat16 *)out_weight,
                              batch, row, col, wsize, eps, dtype);
    } else if (dtype == GGML_TYPE_F16) {
        classify<half>((half *)out, (half *)ff, (const float *)norm_weight, (const half *)out_weight,
                       batch, row, col, wsize, eps, dtype);
    } else {
        classify<float>((float *)out, (float *)ff, (const float *)norm_weight, (const float *)out_weight,
                        batch, row, col, wsize, eps, dtype);
    }
}

void cuda_predict(void *out, void *ff, const void *norm_weight, const void *out_weight, int batch, int row, int col, int wsize, float eps, int dtype)
{
    if (dtype == GGML_TYPE_BF16) {
        predict<nv_bfloat16>((nv_bfloat16 *)out, (nv_bfloat16 *)ff, (const float *)norm_weight, (const nv_bfloat16 *)out_weight,
                             batch, row, col, wsize, eps, dtype);
    } else if (dtype == GGML_TYPE_F16) {
        predict<half>((half *)out, (half *)ff, (const float *)norm_weight, (const half *)out_weight,
                      batch, row, col, wsize, eps, dtype);
    } else {
        predict<float>((float *)out, (float *)ff, (const float *)norm_weight, (const float *)out_weight,
                       batch, row, col, wsize, eps, dtype);
    }
}

} // extern "C"
