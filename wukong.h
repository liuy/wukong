#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <float.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define WARP_SIZE 32U
// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#if defined(ENABLE_BF16)
    typedef __nv_bfloat16 floatX;
    typedef __nv_bfloat16 floatN;
    #define CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
    #define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_16F
#elif defined(ENABLE_FP16)
    typedef half floatX;
    typedef half floatN;
#else
    #define CUBLAS_LOWP CUDA_R_32F // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
    #define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F
    typedef float floatX;
    typedef float floatN;
#endif

#define panic(fmt, ...) do { \
    fprintf(stderr, "%s:%d:%s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
    exit(EXIT_FAILURE); \
    } while (0)

#define cublas_check(status) do { \
    if ((status) != CUBLAS_STATUS_SUCCESS) panic("cublas error: %d", status); \
} while (0)

#define cuda_check(status) do { \
    if ((status) != cudaSuccess) panic("CUDA error: %s", cudaGetErrorString(status)); \
} while (0)

#define cudnn_check(status) do { \
    if (status != CUDNN_STATUS_SUCCESS) panic("CUDNN error: %s\n", cudnnGetErrorString(status)); \
} while (0)

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

static inline float* malloc_rand_float(size_t N)
{
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    return arr;
}

static inline void _printm(const char *name, float* matrix, int batch, int row, int col)
{
    printf("%s[%d:%d:%d]:\n", name, batch, row, col);
    for (int b = 0; b < batch; ++b) {
        printf("Batch %d:\n", b);
        for (int r = 0; r < row; ++r) {
            for (int c = 0; c < col; ++c) {
                int index = b * row * col + r * col + c;
                printf("%f ", matrix[index]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

#define printm(mat, batch, row, col) _printm(#mat, mat, batch, row, col)

extern "C" {
void cuda_init(void);
void cuda_fini(void);
void cuda_to_host(void* dst, void* src, size_t size);
void cuda_to_device(void* dst, void* src, size_t size);
void* cuda_malloc(size_t size);
void cuda_free(void* ptr);
void cuda_matmul(void *out, const void *inp, const void *weight, const void *bias,
            int row, int column, int oc);
void cuda_softmax(void* output, void* intput, int row, int col);
void cuda_mha_attention(void *out, const void *inp, int batch, int row, int NH, int HS);
void cuda_gqa_attention(void *out, const void *inp, int batch, int row, int qNH, int kvNH, int HS);
void cuda_mqa_attention(void *out, const void *inp, int batch, int row, int qNH, int HS);
void cuda_rmsnorm(void* out, const void* inp, const void* weight, int batch, int row, int col);
void cuda_swiglu(void *out, const void *inp, int batch, int row, int col);
void cuda_rope(void *out, const void *inp, const void *freqs_cis, int batch, int row, int NH, int HS);
void cuda_get_freqs_cis(void *freqs_cis, int HS, int row, float theta, int use_scaled);
}

#endif