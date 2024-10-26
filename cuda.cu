#include "wukong.h"

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
static size_t cublaslt_workspace_size = 32 * 1024 * 1024;
static void* cublaslt_workspace = NULL;
static cublasComputeType_t cublas_compute_type;
static cublasLtHandle_t cublaslt_handle;
static cudnnHandle_t cudnn_handle;
static int cuda_arch_major = 0;
static int cuda_arch_minor = 0;
static int cuda_num_SMs = 0; // for persistent threads where we want 1 threadblock per SM
static int cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

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
    printf("CUDA device: %s, major %d, minor %d, num_SMs: %d, threads_per_SM: %d\n",
            deviceProp.name, cuda_arch_major, cuda_arch_minor, cuda_num_SMs, cuda_threads_per_SM);

    cudnn_check(cudnnCreate(&cudnn_handle));
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
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnn_check(cudnnCreateTensorDescriptor(&inputDesc));
    cudnn_check(cudnnCreateTensorDescriptor(&outputDesc));

    cudnn_check(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, row, col, 1, 1));
    cudnn_check(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, row, col, 1, 1));

    float alpha = 1.0f, beta = 0.0f;
    cudnn_check(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
                                    inputDesc, input, &beta, outputDesc, output));

    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
}
}