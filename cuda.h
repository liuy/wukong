void cuda_init(void);
void cuda_fini(void);
void* cuda_malloc(size_t size);
void cuda_to_host(void* dst, void* src, size_t size);
void cuda_to_device(void* dst, void* src, size_t size);
void cuda_free(void* ptr);
void cuda_matmul(void *out, const void *inp, const void *weight, const void *bias, int row, int column, int oc);
void cuda_softmax(void* output, void* input, int row, int col);
void cuda_mha_attention(void *out, const void *inp, int batch, int row, int NH, int HS);
void cuda_gqa_attention(void *out, const void *inp, int batch, int row, int qNH, int kvNH, int HS);
void cuda_mqa_attention(void *out, const void *inp, int batch, int row, int qNH, int HS);
void cuda_rmsnorm(void* out, const void* inp, const void* weight, int N, int col, float eps);
void cuda_swiglu(void *out, const void *inp, int batch, int row, int col);
void cuda_rope(void *out, const void *inp, const void *freqs_cis, int batch, int row, int NH, int HS);
void cuda_get_freqs_cis(void *freqs, int HS, int row, float theta, int use_scaled);
void cuda_embedding(void* out, const void *inp, const void *embd, int batch, int row, int col);
void cuda_cat(void *out, const void *a, const void *b, int arow, int brow, int col);
