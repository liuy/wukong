#include "wukong.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <ctime>

static inline void assert_array_eq(const float *a, const float *b, size_t n)
{
    for (int i = 0; i < n; i++)
        EXPECT_NEAR(a[i], b[i], 1e-5);
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
    float weight[c * oc] = {-0.270431f, 0.026802f, 0.904459f,
                            0.832390f, 0.271423f, 0.434594f,
                            -0.716795f, 0.213938f, -0.967399f,
                            -0.514226f, -0.725537f, 0.608353f};
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

TEST(Cuda, cuda_mha_attention)
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

TEST(Cuda, cuda_rmsnorm)
{
    int batch = 2;
    int row = 2;
    int col = 4;

    float inp[batch * row * col] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f,
        1.3f, 1.4f, 1.5f, 1.6f
    };

    float weight[col] = {0.5f, 0.6f, 0.7f, 0.8f};

    float out[batch * row * col] = {0};

    float res[batch * row * col] = {
        0.182562f, 0.438149f, 0.766761f, 1.168397f,
        0.379045f, 0.545824f, 0.742928f, 0.970354f,
        0.426160f, 0.568214f, 0.729208f, 0.909142f,
        0.446948f, 0.577595f, 0.721993f, 0.880144f,
    };

    void *d_out = cuda_malloc(batch * row * col * sizeof(float));
    void *d_inp = cuda_malloc(batch * row * col * sizeof(float));
    void *d_weight = cuda_malloc(col * sizeof(float));

    cuda_to_device(d_inp, inp, batch * row * col * sizeof(float));
    cuda_to_device(d_weight, weight, col * sizeof(float));

    cuda_rmsnorm(d_out, d_inp, d_weight, batch, row, col);
    cuda_to_host(out, d_out, batch * row * col * sizeof(float));

    assert_array_eq(res, out, batch * row * col);

    cuda_free(d_out);
    cuda_free(d_inp);
    cuda_free(d_weight);
}

TEST(Cuda, cuda_swiglu)
{
    int batch = 2;
    int row = 2;
    int col = 3;
    int hidden_size = 4;

    float inp[batch * row * col] = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f
    };

    // Expected output tensor
    float res[batch * row * hidden_size] = {
        0.010485f, 0.059323f, 0.155615f, 0.306913f,
        0.059323f, 0.405260f, 1.149139f, 2.347071f,
        0.010485f, 0.059323f, 0.155615f, 0.306913f,
        0.059323f, 0.405260f, 1.149139f, 2.347071f
    };

    // concatenate the weights
    float weights_fc[2 * hidden_size * col] = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f,
        1.0f, 1.1f, 1.2f,
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f,
        1.0f, 1.1f, 1.2f
    };
    void *d_out = cuda_malloc(batch * row * hidden_size * sizeof(float));
    void *d_inp = cuda_malloc(batch * row * col * sizeof(float));
    void *d_fcout = cuda_malloc(batch * row * 2 * hidden_size * sizeof(float));
    void *d_weights_fc = cuda_malloc(2 * hidden_size * col * sizeof(float));

    cuda_to_device(d_inp, inp, batch * row * col * sizeof(float));
    cuda_to_device(d_weights_fc, weights_fc, 2 * hidden_size * col * sizeof(float));

    cuda_matmul(d_fcout, d_inp, d_weights_fc, nullptr, batch * row, col, 2 * hidden_size);
    cuda_swiglu(d_out, d_fcout, batch, row, hidden_size);

    float out[batch * row * hidden_size] = {0};
    cuda_to_host(out, d_out, batch * row * hidden_size * sizeof(float));

    assert_array_eq(res, out, batch * row * hidden_size);

    cuda_free(d_out);
    cuda_free(d_inp);
    cuda_free(d_fcout);
    cuda_free(d_weights_fc);
}

void get_freqs_cis(floatX *freqs_cis, int dim, int end, float theta, int use_scaled)
{
    // helper function that (on the CPU!) precomputes the freqs_cis for the RoPE rotation
    // same as precompute_freqs_cis_real in rope.py
    for (int i = 0; i < dim / 2; i++) {

        // calculate the frequency for the (i, i+1)th dimension
        float freq = 1.0f / powf(theta, (float)(2 * i) / dim);
        if (use_scaled) {
            const int scale_factor = 8;
            const int low_freq_factor = 1;
            const int high_freq_factor = 4;
            const int old_context_len = 8192;  // original llama3 length
            const float low_freq_wavelen = (float)old_context_len / low_freq_factor;
            const float high_freq_wavelen = (float)old_context_len / high_freq_factor;
            float wavelen = 2.0f * M_PI / freq;
            if (wavelen < high_freq_wavelen) {
                // skip; keep freq as is
            } else if (wavelen > low_freq_wavelen) {
                // scale down by scale_factor
                freq /= scale_factor;
            } else {
                // smooth transition between scaled and unscaled
                float smooth = ((float)old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
                freq = (1.0f - smooth) * freq / scale_factor + smooth * freq;
            }
        }

        // iterate over all time steps, calculate the angle, and store the cos/sin
        for (int t = 0; t < end; t++) {
            float angle = (float)t * freq;
            freqs_cis[t * dim + 2 * i] = cosf(angle);     // real part
            freqs_cis[t * dim + 2 * i + 1] = sinf(angle); // imaginary part
        }
    }
}

TEST(Cuda, cuda_rope)
{
    int batch = 1;
    int row = 2;
    int NH = 2;
    int HS = 2;

    float qkv[batch * row * 3 * NH * HS] = {
        // row 0
        0.1f, 0.2f, 0.3f, 0.4f, // q
        0.5f, 0.6f, 0.7f, 0.8f, // k
        0.9f, 1.0f, 1.1f, 1.2f, // v
        // row 1
        1.3f, 1.4f, 1.5f, 1.6f, // q
        1.7f, 1.8f, 1.9f, 2.0f, // k
        2.1f, 2.2f, 2.3f, 2.4f, // v
    };
    float fc[HS*row] = {0};
    float fc_res[HS*row] = {
        1.000000f, 0.000000f,
        0.540302f, 0.841471f,
    };
    float out[batch * row * 3 * NH * HS] = {0};
    float res[batch * row * 3 * NH * HS] = {
        0.100000f, 0.200000f, 0.300000f, 0.400000f,
        0.500000f, 0.600000f, 0.700000f, 0.800000f,
        0.900000f, 1.000000f, 1.100000f, 1.200000f,
        -0.475666f, 1.850335f, -0.535900f, 2.126690f,
        -0.596134f, 2.403045f, -0.656368f, 2.679399f,
        2.100000f, 2.200000f, 2.300000f, 2.400000f
    };

    get_freqs_cis(fc, HS, row, 10000.0f, false);
    assert_array_eq(fc_res, fc, HS * row);

    void *d_qkv = cuda_malloc(batch * row * 3 * NH * HS * sizeof(float));
    void *d_fc = cuda_malloc(HS * row * sizeof(float));

    cuda_to_device(d_qkv, qkv, batch * row * 3 * NH * HS * sizeof(float));
    cuda_to_device(d_fc, fc, HS * row * sizeof(float));
    cuda_rope(d_qkv, d_qkv, d_fc, batch, row, NH, HS); // update qkv in-place
    cuda_to_host(out, d_qkv, batch * row * 3 * NH * HS * sizeof(float));
    assert_array_eq(res, out, batch * row * 3 * NH * HS);

    float fcs[HS*row] = {0};
    float fcs_res[HS*row] = {
        1.000000f, 0.000000f,
        0.540302f, 0.841471f,
    };
    get_freqs_cis(fcs, HS, row, 10000.0f, true);
    assert_array_eq(fcs_res, fcs, HS * row);

    cuda_free(d_qkv);
    cuda_free(d_fc);
}

TEST(Cuda, cuda_get_freqs_cis)
{
    std::srand(std::time(nullptr));
    for (int i = 0; i < 10; ++i) {
        int HS = (std::rand() % 40 + 1) * 2; // random even number between 2 and 80
        int row = (std::rand() % 40 + 1) * 2; // random even number between 2 and 80
        float theta = static_cast<float>(std::rand() % 10000 + 1); // random value between 1 and 10000

        float res[HS*row] = {0};
        float out[HS*row] = {0};

        void *d_out = cuda_malloc(HS * row * sizeof(float));
        cuda_get_freqs_cis(d_out, HS, row, theta, false);
        cuda_to_host(out, d_out, HS * row * sizeof(float));
        get_freqs_cis(res, HS, row, theta, false);
        assert_array_eq(res, out, HS * row);

        cuda_get_freqs_cis(d_out, HS, row, theta, true);
        cuda_to_host(out, d_out, HS * row * sizeof(float));
        get_freqs_cis(res, HS, row, theta, true);
        assert_array_eq(res, out, HS * row);

        cuda_free(d_out);
    }
}