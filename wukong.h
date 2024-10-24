#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cudnn.h>

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
void matmul(float *out, const float *inp, const float *weight, const float *bias,
            int row, int column, int oc);
void softmax(float* input, float* output, int row, int col);
}

#endif