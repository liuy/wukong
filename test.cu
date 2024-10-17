#include "common.h"
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

TEST(Cuda, matmul)
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
    matmul(out, inp, weight, bias, b, r, c, oc);
    assert_array_eq(res, out, b * r * oc);
    matmul(out, inp, weight, nullptr, b, r, c, oc);
    assert_array_eq(res_nob, out, b * r * oc);
}