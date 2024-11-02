#include "wukong.h"
#include <gtest/gtest.h>

static inline void assert_array_eq(const float *a, const float *b, size_t n)
{
    for (int i = 0; i < n; i++)
        EXPECT_NEAR(a[i], b[i], 1e-6);
}

class cudaEnv : public ::testing::Environment {
public:
  void SetUp() override { cuda_init(); }
  void TearDown() override { cuda_fini(); }
};

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new cudaEnv);
    return RUN_ALL_TESTS();
}

TEST(Cuda, cuda_matmul)
{
    int b = 2;
    int r = 2;
    int c = 3;
    int oc = 4;

    float out[b * r * oc] = {0};
    float inp[b * r * c] = {0.680375f, -0.211234f, 0.566198f,
                            0.596880f, 0.823295f, -0.604897f,
                            -0.329554f, 0.536459f, -0.444451f,
                            0.107940f, -0.045206f, 0.257742f};
    float weight[c * oc] = {-0.270431f, 0.832390f, -0.716795f, -0.514226f,
                            0.026802f, 0.271423f, 0.213938f, -0.725537f,
                            0.904459f, 0.434594f, -0.967399f, 0.608353f };
    float bias[oc] = {0.1f, 0.2f, 0.3f, 0.4f};
    float res[b * r * oc] = {0.422447f, 0.955070f, -0.780620f, 0.547840f,
                             -0.586453f, 0.657414f, 0.633470f, -0.872253f,
                             -0.198488f, -0.121866f, 1.080953f, -0.090139f,
                             0.302715f, 0.389591f, -0.036381f, 0.534091f};
    float res_nob[] = {0.322447f, 0.755070f, -1.080620f, 0.147840f,
                       -0.686453f, 0.457414f, 0.333470f, -1.272253f,
                       -0.298488f, -0.321866f, 0.780953f, -0.490139f,
                       0.202715f, 0.189591f, -0.336381f, 0.134091f};
    void *d_out = cuda_malloc(b * r * oc * sizeof(float));
    void *d_inp = cuda_malloc(b * r * c * sizeof(float));
    void *d_weight = cuda_malloc(c * oc * sizeof(float));
    void *d_bias = cuda_malloc(oc * sizeof(float));
    cuda_to_device(d_inp, inp, b * r * c * sizeof(float));
    cuda_to_device(d_weight, weight, c * oc * sizeof(float));
    cuda_to_device(d_bias, bias, oc * sizeof(float));

    cuda_matmul(d_out, d_inp, d_weight, d_bias, b * r, c, oc);
    cuda_to_host(out, d_out, b * r * oc * sizeof(float));
    assert_array_eq(res, out, b * r * oc);
    cuda_matmul(d_out, d_inp, d_weight, nullptr, b * r, c, oc);
    cuda_to_host(out, d_out, b * r * oc * sizeof(float));
    assert_array_eq(res_nob, out, b * r * oc);

    cuda_free(d_out);
    cuda_free(d_inp);
    cuda_free(d_weight);
    cuda_free(d_bias);
}

TEST(Cuda, cuda_softmax)
{
    float inp[2 * 3] = {2.0f, 2.0f, 2.0f, 4.0f, 1000.0f, 1.0f};
    float out[2 * 3] = {0};
    float res[] = {0.333333f, 0.333333f, 0.333333f, 0.000000f, 1.000000f, 0.000000f};

    void *d_out = cuda_malloc(2 * 3 * sizeof(float));
    void *d_inp = cuda_malloc(2 * 3 * sizeof(float));
    cuda_to_device(d_inp, inp, 2 * 3 * sizeof(float));
    cuda_softmax(d_out, d_inp, 2, 3);
    cuda_to_host(out, d_out, 2 * 3 * sizeof(float));
    assert_array_eq(res, out, 6);
    cuda_free(d_out);
    cuda_free(d_inp);
}

TEST(Cuda, cuda_flash_attention)
{
    int batch = 2;
    int row = 4;
    int NH = 1;
    int HS = 2;
    int col = NH * HS;

    float inp[batch * row * col * 3] = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, // Batch1
        0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f,
        1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f,
        1.9f, 2.0f, 2.1f, 2.2f, 2.3f, 2.4f,
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, // Batch2
        0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f,
        1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f,
        1.9f, 2.0f, 2.1f, 2.2f, 2.3f, 2.4f,
    };

    float out[batch * row * col] = {0};
    float res[batch * row * col] = {
        0.500000f, 0.600000f,
        0.892363f, 0.992363f,
        1.479998f, 1.579998f,
        2.161403f, 2.261404f,
        0.500000f, 0.600000f,
        0.892363f, 0.992363f,
        1.479998f, 1.579998f,
        2.161403f, 2.261404f,
    };

    void *d_out = cuda_malloc(batch * row * col * sizeof(float));
    void *d_inp = cuda_malloc(batch * row * col * 3 * sizeof(float));
    cuda_to_device(d_inp, inp, batch * row * col * 3 * sizeof(float));

    cuda_mha_attention(d_out, d_inp, batch, row, NH, HS);
    cuda_to_host(out, d_out, batch * row * col * sizeof(float));
    assert_array_eq(res, out, batch * row * col);
    // printm(out, batch, row, col);

    cuda_free(d_out);
    cuda_free(d_inp);
}

TEST(Cuda, cuda_gqa_attention)
{
    { // Case: Single element input
        int batch = 1;
        int row = 1;
        int qNH = 1;
        int kvNH = 1;
        int HS = 1;

        float inp[3] = {0.1f, 0.2f, 0.3f};
        float out[1] = {0};

        float res[1] = {0.3f};

        void *d_out = cuda_malloc(1 * sizeof(float));
        void *d_inp = cuda_malloc(3 * sizeof(float));
        cuda_to_device(d_inp, inp, 3 * sizeof(float));

        cuda_gqa_attention(d_out, d_inp, batch, row, qNH, kvNH, HS);
        cuda_to_host(out, d_out, 1 * sizeof(float));
        assert_array_eq(res, out, 1);

        cuda_free(d_out);
        cuda_free(d_inp);
    }

    {
        int batch = 2;
        int row = 2;
        int qNH = 4;
        int kvNH = 2;
        int HS = 2;
        int qSize = qNH * HS;
        int kvSize = kvNH * HS;

        float inp[batch * row * (qSize + 2*kvSize)] = {
            // Batch 1, Row 1
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,     // Q (4 heads * 2 dims)
            0.1f, 0.2f, 0.3f, 0.4f,                             // K (2 heads * 2 dims)
            0.5f, 0.6f, 0.7f, 0.8f,                             // V (2 heads * 2 dims)
            // Batch 1, Row 2
            1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f,     // Q
            0.9f, 1.0f, 1.1f, 1.2f,                             // K
            1.3f, 1.4f, 1.5f, 1.6f,                             // V
            // Batch 2, Row 1 (same pattern as batch 1)
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
            0.1f, 0.2f, 0.3f, 0.4f,
            0.5f, 0.6f, 0.7f, 0.8f,
            // Batch 2, Row 2
            1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f,
            0.9f, 1.0f, 1.1f, 1.2f,
            1.3f, 1.4f, 1.5f, 1.6f
        };

        float out[batch * row * qSize] = {0};

        float res[batch * row * qSize] = {
            // Batch 0
            0.500000f, 0.600000f, 0.700000f, 0.800000f, 0.500000f, 0.600000f, 0.700000f, 0.800000f,
            1.128813f, 1.228813f, 1.357295f, 1.457295f, 1.181927f, 1.281927f, 1.402936f, 1.502936f,
            // Batch 1
            0.500000f, 0.600000f, 0.700000f, 0.800000f, 0.500000f, 0.600000f, 0.700000f, 0.800000f,
            1.128813f, 1.228813f, 1.357295f, 1.457295f, 1.181927f, 1.281927f, 1.402936f, 1.502936f
        };

        void *d_out = cuda_malloc(batch * row * qSize * sizeof(float));
        void *d_inp = cuda_malloc(batch * row * (qSize + 2*kvSize) * sizeof(float));
        cuda_to_device(d_inp, inp, batch * row * (qSize + 2*kvSize) * sizeof(float));

        cuda_gqa_attention(d_out, d_inp, batch, row, qNH, kvNH, HS);
        cuda_to_host(out, d_out, batch * row * qSize * sizeof(float));
        assert_array_eq(res, out, batch * row * qSize);

        cuda_free(d_out);
        cuda_free(d_inp);
    }
}

TEST(Cuda, cuda_mqa_attention)
{
    int batch = 2;
    int row = 2;
    int qNH = 4;
    int HS = 2;
    int qSize = qNH * HS;
    int kvSize = HS; // kvNH = 1 for MQA

    float inp[batch * row * (qSize + 2*kvSize)] = {
        // Batch 1, Row 1
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,     // Q (4 heads * 2 dims)
        0.1f, 0.2f,                                          // K (1 head * 2 dims)
        0.5f, 0.6f,                                          // V (1 head * 2 dims)
        // Batch 1, Row 2
        1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f,     // Q
        0.9f, 1.0f,                                          // K
        1.3f, 1.4f,                                          // V
        // Batch 2, Row 1 (same pattern as batch 1)
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.1f, 0.2f,
        0.5f, 0.6f,
        // Batch 2, Row 2
        1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f,
        0.9f, 1.0f,
        1.3f, 1.4f
    };

    float out[batch * row * qSize] = {0};

    float res[batch * row * qSize] = {
        // Batch 0
        0.500000f, 0.600000f, 0.500000f, 0.600000f, 0.500000f, 0.600000f, 0.500000f, 0.600000f,
        1.128813f, 1.228813f, 1.157295f, 1.257295f, 1.181927f, 1.281927f, 1.202936f, 1.302936f,
        // Batch 1
        0.500000f, 0.600000f, 0.500000f, 0.600000f, 0.500000f, 0.600000f, 0.500000f, 0.600000f,
        1.128813f, 1.228813f, 1.157295f, 1.257295f, 1.181927f, 1.281927f, 1.202936f, 1.302936f
    };

    void *d_out = cuda_malloc(batch * row * qSize * sizeof(float));
    void *d_inp = cuda_malloc(batch * row * (qSize + 2*kvSize) * sizeof(float));
    cuda_to_device(d_inp, inp, batch * row * (qSize + 2*kvSize) * sizeof(float));

    cuda_mqa_attention(d_out, d_inp, batch, row, qNH, HS);
    cuda_to_host(out, d_out, batch * row * qSize * sizeof(float));
    assert_array_eq(res, out, batch * row * qSize);

    cuda_free(d_out);
    cuda_free(d_inp);
}