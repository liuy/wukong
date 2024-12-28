void cuda_init(void);
void cuda_fini(void);
void* cuda_malloc(size_t size);
void cuda_to_host(void* dst, void* src, size_t size);
void cuda_to_device(void* dst, void* src, size_t size);
void cuda_free(void* ptr);
void cuda_matmul(void *out, const void *inp, const void *weight, const void *bias, int row, int column, int oc, int dtype);
void cuda_softmax(void* output, void* input, int row, int col);
void cuda_mh_sdpa(void *out, const void *inp, int batch, int row, int NH, int HS);
void cuda_gq_sdpa(void *out, const void *inp, int batch, int row, int qNH, int kvNH, int HS);
void cuda_mq_sdpa(void *out, const void *inp, int batch, int row, int qNH, int HS);
void cuda_rmsnorm(void *out, const void *inp, const void *weight, int row, int col, float eps);
void cuda_swiglu(void *out, const void *inp, int batch, int row, int col);
void cuda_rope_qkv(void *out, const void *inp, const void *raw_freqs, int batch, int row, int NH, int kvNH, int HS);
void cuda_rope(void *out, const void *inp, const void *raw_freqs, int batch, int row, int NH, int HS);
void cuda_get_freqs_cis(void *freqs, int HS, int row, float theta, int use_scaled);
void cuda_embedding(void* out, const void *inp, const void *embd, int batch, int row, int col, int dtype);
void cuda_cat(void *out, const void *a, const void *b, int arow, int brow, int col);
void cuda_div(void *out, const void *a, const void *b, int row, int col);
void cuda_dequantize(void *out, const void *inp, int row, int col, int type);
void cuda_add(void* out, const void* a, const void* b, int row, int col);
void cuda_group_query_attention(void *out, const void *embeds, const void *freqs, const void *out_weight, const void *norm_weight, const void *qkv_weight, int batch, int row, int NH, int kvNH, int HS, float eps, int dtype);
