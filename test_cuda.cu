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

    cuda_rmsnorm(d_out, d_inp, d_weight, batch * row, col, 1e-5);
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

// Precompute the freqs of the RoPE rotation for the given HS(HeadSize) and theta
// return array of size HS/2
void get_freqs(floatX *freqs, int HS, float theta)
{
    // helper function that (on the CPU!) precomputes the freqs_cis for the RoPE rotation
    // same as precompute_freqs_cis_real in rope.py
    for (int i = 0; i < HS / 2; i++) {

        // calculate the frequency for the (i, i+1)th dimension
        float freq = 1.0f / powf(theta, (float)(2 * i) / HS);
        const int scale_factor = 8;
        const int low_freq_factor = 1;
        const int high_freq_factor = 4;
        const int old_context_len = 8192;  // original llama3 length
        const float low_freq_wavelen = (float)old_context_len / low_freq_factor;
        const float high_freq_wavelen = (float)old_context_len / high_freq_factor;
        float wavelen = 2.0f * M_PI / freq;
        if (wavelen < high_freq_wavelen) {
            // skip; keep freq as is
        }
        else if (wavelen > low_freq_wavelen) {
            // scale down by scale_factor
            freq /= scale_factor;
        }
        else {
            // smooth transition between scaled and unscaled
            float smooth = ((float)old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            freq = (1.0f - smooth) * freq / scale_factor + smooth * freq;
        }
        freqs[i] = freq;
    }
}

// return array of size row * HS
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
    int batch = 2;
    int row = 3;
    int NH = 2;
    int HS = 2;

    float qkv[batch * row * NH * HS] = {
        // batch 0
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f,
        // batch 1
        1.3f, 1.4f, 1.5f, 1.6f,
        1.7f, 1.8f, 1.9f, 2.0f,
        2.1f, 2.2f, 2.3f, 2.4f,
    };
    float fc[HS/2] = {0};
    float fc_res[HS/2] = {
        1.000000f,
    };
    float out[batch * row * NH * HS] = {0};
    float res[batch * row * NH * HS] = {
        0.100000f, 0.200000f, 0.30000f, 0.400000f,
        -2.347315e-01, 7.449169e-01, -2.949652e-01, 1.021272e+00,
        -1.283830e+00, 4.022209e-01, -1.548918e+00, 5.008510e-01,
        1.300000e+00, 1.400000e+00, 1.500000e+00, 1.600000e+00,
        -0.596134f, 2.403045f, -0.656368f, 2.679399f,
        -2.874363e+00, 9.940016e-01, -3.139452e+00, 1.092632e+00
    };

    get_freqs(fc, HS, 10000.0f);
    assert_array_eq(fc_res, fc, HS / 2);

    void *d_qkv = cuda_malloc(batch * row * NH * HS * sizeof(float));
    void *d_fc = cuda_malloc(HS / 2 * sizeof(float));

    cuda_to_device(d_qkv, qkv, batch * row * NH * HS * sizeof(float));
    cuda_to_device(d_fc, fc, HS / 2 * sizeof(float));
    cuda_rope(d_qkv, d_qkv, d_fc, batch, row, NH, HS); // update qkv in-place
    cuda_to_host(out, d_qkv, batch * row * NH * HS * sizeof(float));
    assert_array_eq(res, out, batch * row * NH * HS);

    cuda_free(d_qkv);
    cuda_free(d_fc);

    NH = 1;
    HS = 4;
    float fc_res2[HS/2] = {
        1.000000f, 0.010000f,
    };
    float res2[batch * row * 3 * NH * HS] = {
        0.100000f, 0.200000f, 0.300000f, 0.400000f,
        -2.347315e-01, 7.449169e-01, 6.919651e-01, 8.069599e-01,
        -1.283830e+00, 4.022209e-01, 1.075782e+00, 1.221759e+00,
        1.300000e+00, 1.400000e+00, 1.500000e+00, 1.600000e+00,
        -5.961339e-01, 2.403045e+00, 1.879905e+00, 2.018900e+00,
        -2.874363e+00, 9.940016e-01, 2.251543e+00, 2.445517e+00,
    };
    get_freqs(fc, HS, 10000.0f);
    assert_array_eq(fc_res2, fc, HS / 2);
    d_qkv = cuda_malloc(batch * row * NH * HS * sizeof(float));
    d_fc = cuda_malloc(HS / 2 * sizeof(float));
    cuda_to_device(d_qkv, qkv, batch * row * NH * HS * sizeof(float));
    cuda_to_device(d_fc, fc, HS / 2 * sizeof(float));
    cuda_rope(d_qkv, d_qkv, d_fc, batch, row, NH, HS); // update qkv in-place
    cuda_to_host(out, d_qkv, batch * row * NH * HS * sizeof(float));
    assert_array_eq(res2, out, batch * row * NH * HS);

    cuda_free(d_qkv);
    cuda_free(d_fc);
}

TEST(Cuda, cuda_embedding) {
    int batch = 2;
    int row = 4;
    int col = 4;
    int vocab_size = 6;

    int inp[] = {0, 2, 1, 3, 4, 5, 1, 0};
    float embd[] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.4f, 0.5f, 0.6f, 0.7f,
        0.7f, 0.8f, 0.9f, 1.0f,
        1.0f, 1.1f, 1.2f, 1.3f,
        1.1f, 1.2f, 1.3f, 1.4f,
        1.4f, 1.5f, 1.6f, 1.7f,
    };
    float h_out[batch * row * col];

    int *d_inp;
    float *d_embd, *d_out;

    cudaMalloc(&d_inp, sizeof(int) * batch * row);
    cudaMalloc(&d_embd, sizeof(float) * vocab_size * col);
    cudaMalloc(&d_out, sizeof(float) * batch * row * col);

    cudaMemcpy(d_inp, inp, sizeof(int) * batch * row, cudaMemcpyHostToDevice);
    cudaMemcpy(d_embd, embd, sizeof(float) * vocab_size * col, cudaMemcpyHostToDevice);

    cuda_embedding(d_out, d_inp, d_embd, batch, row, col);

    cudaMemcpy(h_out, d_out, sizeof(float) * batch * row * col, cudaMemcpyDeviceToHost);

    float expected[] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.7f, 0.8f, 0.9f, 1.0f,
        0.4f, 0.5f, 0.6f, 0.7f,
        1.0f, 1.1f, 1.2f, 1.3f,
        1.1f, 1.2f, 1.3f, 1.4f,
        1.4f, 1.5f, 1.6f, 1.7f,
        0.4f, 0.5f, 0.6f, 0.7f,
        0.1f, 0.2f, 0.3f, 0.4f,
    };
    assert_array_eq(h_out, expected, batch * row * col);

    cudaFree(d_inp);
    cudaFree(d_embd);
    cudaFree(d_out);
}
TEST(Cuda, cuda_cat) {
    int arow = 2;
    int brow = 1;
    int col = 3;

    float a[arow * col] = {1.0f, 2.0f, 3.0f,
                           4.0f, 5.0f, 6.0f};
    float b[brow * col] = {7.0f, 8.0f, 9.0f};
    float out[(arow + brow) * col] = {0};

    float expected[(arow + brow) * col] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    };

    void *d_out = cuda_malloc((arow + brow) * col * sizeof(float));
    void *d_a = cuda_malloc(arow * col * sizeof(float));
    void *d_b = cuda_malloc(brow * col * sizeof(float));

    cuda_to_device(d_a, a, arow * col * sizeof(float));
    cuda_to_device(d_b, b, brow * col * sizeof(float));

    cuda_cat(d_out, d_a, d_b, arow, brow, col);
    cuda_to_host(out, d_out, (arow + brow) * col * sizeof(float));

    assert_array_eq(out, expected, (arow + brow) * col);

    cuda_free(d_out);
    cuda_free(d_a);
    cuda_free(d_b);
}

TEST(Cuda, cuda_div)
{
    int row = 2;
    int col = 3;

    float a[row * col] = {6.0f, 12.0f, 18.0f,
                          24.0f, 30.0f, 36.0f};
    float b[row * col] = {2.0f, 3.0f, 6.0f,
                          8.0f, 5.0f, 9.0f};
    float out[row * col] = {0};

    float expected[row * col] = {
        3.0f, 4.0f, 3.0f,
        3.0f, 6.0f, 4.0f,
    };

    void *d_out = cuda_malloc(row * col * sizeof(float));
    void *d_a = cuda_malloc(row * col * sizeof(float));
    void *d_b = cuda_malloc(row * col * sizeof(float));

    cuda_to_device(d_a, a, row * col * sizeof(float));
    cuda_to_device(d_b, b, row * col * sizeof(float));

    cuda_div(d_out, d_a, d_b, row, col);
    cuda_to_host(out, d_out, row * col * sizeof(float));

    assert_array_eq(out, expected, row * col);

    cuda_free(d_out);
    cuda_free(d_a);
    cuda_free(d_b);
}

TEST(Cuda, cuda_dequantize)
{
    int row = 2;
    int nb = 2;
    int col = nb * 32;
    int type = GGML_TYPE_Q8_0;

    uint8_t inp[row * nb * 34] = {
        8, 52, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127,
        0, 25, 0, 252, 8, 244, 16, 236, 25, 227, 33, 219, 41, 211, 49, 203, 57, 195, 66, 186, 74, 178, 82, 170, 90, 162, 98, 154, 107, 145, 115, 137, 123, 129,
        0, 12, 0, 4, 8, 12, 16, 20, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 107, 111, 115, 119, 123, 127,
        154, 1, 0, 4, 8, 12, 16, 20, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 107, 111, 115, 119, 123, 127,
    };

    float out[row * col] = {0};

    float expected[row * col] = {
        1.007812e+00, 2.015625e+00, 3.023438e+00, 4.031250e+00, 5.039062e+00, 6.046875e+00, 7.054688e+00, 8.062500e+00, 9.070312e+00, 1.007812e+01, 1.108594e+01, 1.209375e+01, 1.310156e+01, 1.410938e+01, 1.511719e+01, 1.612500e+01, 1.688086e+01, 1.788867e+01, 1.889648e+01, 1.990430e+01, 2.091211e+01, 2.191992e+01, 2.292773e+01, 2.393555e+01, 2.494336e+01, 2.595117e+01, 2.695898e+01, 2.796680e+01, 2.897461e+01, 2.998242e+01, 3.099023e+01, 3.199805e+01,
        0.000000e+00, -9.765625e-03, 1.953125e-02, -2.929688e-02, 3.906250e-02, -4.882812e-02, 6.103516e-02, -7.080078e-02, 8.056641e-02, -9.033203e-02, 1.000977e-01, -1.098633e-01, 1.196289e-01, -1.293945e-01, 1.391602e-01, -1.489258e-01, 1.611328e-01, -1.708984e-01, 1.806641e-01, -1.904297e-01, 2.001953e-01, -2.099609e-01, 2.197266e-01, -2.294922e-01, 2.392578e-01, -2.490234e-01, 2.612305e-01, -2.709961e-01, 2.807617e-01, -2.905273e-01, 3.002930e-01, -3.100586e-01,
        0.000000e+00, 9.765625e-04, 1.953125e-03, 2.929688e-03, 3.906250e-03, 4.882812e-03, 6.103516e-03, 7.080078e-03, 8.056641e-03, 9.033203e-03, 1.000977e-02, 1.098633e-02, 1.196289e-02, 1.293945e-02, 1.391602e-02, 1.489258e-02, 1.611328e-02, 1.708984e-02, 1.806641e-02, 1.904297e-02, 2.001953e-02, 2.099609e-02, 2.197266e-02, 2.294922e-02, 2.392578e-02, 2.490234e-02, 2.612305e-02, 2.709961e-02, 2.807617e-02, 2.905273e-02, 3.002930e-02, 3.100586e-02,
        0.000000e+00, 9.775162e-05, 1.955032e-04, 2.932549e-04, 3.910065e-04, 4.887581e-04, 6.109476e-04, 7.086992e-04, 8.064508e-04, 9.042025e-04, 1.001954e-03, 1.099706e-03, 1.197457e-03, 1.295209e-03, 1.392961e-03, 1.490712e-03, 1.612902e-03, 1.710653e-03, 1.808405e-03, 1.906157e-03, 2.003908e-03, 2.101660e-03, 2.199411e-03, 2.297163e-03, 2.394915e-03, 2.492666e-03, 2.614856e-03, 2.712607e-03, 2.810359e-03, 2.908111e-03, 3.005862e-03, 3.103614e-03
    };

    void *d_out = cuda_malloc(row * col * sizeof(float));
    void *d_inp = cuda_malloc(row * 2 * sizeof(block_q8_0));

    cuda_to_device(d_inp, inp, row * 2 * sizeof(block_q8_0));

    cuda_dequantize(d_out, d_inp, row, col, type);
    cuda_to_host(out, d_out, row * col * sizeof(float));
    assert_array_eq(expected, out, row * col);

    cuda_free(d_out);
    cuda_free(d_inp);
}