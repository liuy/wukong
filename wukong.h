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

// ----------------------------------------------------------------------------
// Packed128 data structure that forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.
template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;
    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    __device__  static Packed128 constant(ElementType value) {
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }
    __device__ static Packed128 zeros() {
        return constant(0.f);
    }
    __device__ static Packed128 ones() {
        return constant(1.f);
    }

    __device__ ElementType& operator[](int index) {
        return payload[index];
    }
    __device__ const ElementType& operator[](int index) const {
        return payload[index];
    }
    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
};

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

// short-form typedefs
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

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
                printf("%e ", matrix[index]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

#define printm(mat, batch, row, col) _printm(#mat, mat, batch, row, col)

extern "C" {
    #include "cuda.h"
}

#endif