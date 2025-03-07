#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <float.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

enum dtype {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_2,  // support has been removed
    GGML_TYPE_Q4_3,  // support has been removed
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_1,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_K,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_I64,
    GGML_TYPE_F64,
    GGML_TYPE_IQ1_M,
    GGML_TYPE_BF16,
    GGML_TYPE_Q4_0_4_4,
    GGML_TYPE_Q4_0_4_8,
    GGML_TYPE_Q4_0_8_8,
    GGML_TYPE_TQ1_0,
    GGML_TYPE_TQ2_0,
    GGML_TYPE_COUNT,
};

struct dtype_info {
    const char *name;
    int block_size;
    int type_size;
};

extern dtype_info dtype_infos[GGML_TYPE_COUNT];

// Q8_0 block layout in byte [s0, s1, d0, d1, ..., d31]
// scale = [s0, s1].astype(float16)
// dequantized block = scale * [d0, d1, ..., d31]
struct block_q8_0 {
    half scale; // 2 bytes for scale
    int8_t d[32];  // 32 bytes for data in a block
};

#define WARP_SIZE 32U

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.
template<class ElementType>
struct alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
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
        return constant(0);
    }

    __device__ static Packed128 ones() {
        return constant(1);
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
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    static constexpr const int size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];

    __device__ Packed128& operator+(const Packed128& other) {
        for (int i = 0; i < size; ++i) {
            payload[i] += other.payload[i];
        }
        return *this;
    }
    __device__ Packed128& operator-(const Packed128& other) {
        for (int i = 0; i < size; ++i) {
            payload[i] -= other.payload[i];
        }
        return *this;
    }
    __device__ Packed128& operator*(const Packed128& other) {
        for (int i = 0; i < size; ++i) {
            payload[i] *= other.payload[i];
        }
        return *this;
    }
    __device__ Packed128& operator/(const Packed128& other) {
        for (int i = 0; i < size; ++i) {
            payload[i] /= other.payload[i];
        }
        return *this;
    }
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
// store a Packed128 to an aligned memory address while cacheing in L1 and L2
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address while bypassing L1 and L2 caches
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

#define panic(fmt, ...) do { \
    fprintf(stderr, "%s:%d:%s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
    exit(EXIT_FAILURE); \
    } while (0)

#define cublas_check(status) do { \
    if ((status) != CUBLAS_STATUS_SUCCESS) panic("cublas error: %d", status); \
} while (0)

#define cuda_check(status) do { \
    if ((status) != cudaSuccess) panic("CUDA error (%d): %s", status, cudaGetErrorString(status)); \
} while (0)

#define cudnn_check(status) do { \
    if (status != CUDNN_STATUS_SUCCESS) panic("CUDNN error: %s\n", cudnnGetErrorString(status)); \
} while (0)

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

static __device__ __forceinline__ nv_bfloat16 f32_to_bf16(float f) {
    return __float2bfloat16(f);
}

static __device__ __forceinline__ float bf16_to_f32(nv_bfloat16 f) {
    return __bfloat162float(f);
}

static __device__ __forceinline__ half f32_to_f16(float f) {
    return __float2half(f);
}

static __device__ __forceinline__ float f16_to_f32(half f) {
    return __half2float(f);
}

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
    fflush(stdout);
}

#define printm(mat, batch, row, col) _printm(#mat, mat, batch, row, col)

static inline void _printc(const char *name, float* matrix, int batch, int row, int col, int c)
{
    printf("Column %d of %s[%d:%d:%d]:\n", c, name, batch, row, col);
    if (c < 0)
        c += col;
    if (c >= col)
        c %= col;

    for (int b = 0; b < batch; ++b) {
        printf("Batch %d:\n", b);
        for (int r = 0; r < row; ++r) {
            int index = b * row * col + r * col + c;
            printf("%e \n", matrix[index]);
        }
        printf("\n");
    }
    fflush(stdout);
}

static inline void _printr(const char *name, float* matrix, int batch, int row, int col, int r)
{
    printf("Row %d of %s[%d:%d:%d]:\n", r, name, batch, row, col);
    if (r < 0)
        r += row;
    if (r >= row)
        r %= row;

    for (int b = 0; b < batch; ++b) {
        printf("Batch %d:\n", b);
        for (int c = 0; c < col; ++c) {
            int index = b * row * col + r * col + c;
            printf("%e ", matrix[index]);
        }
        printf("\n");
    }
    fflush(stdout);
}

#define printc(mat, batch, row, col, c) _printc(#mat, mat, batch, row, col, c)
#define printr(mat, batch, row, col, r) _printr(#mat, mat, batch, row, col, r)

extern "C" {
    #include "cuda.h"
}

#endif
