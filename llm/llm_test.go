package llm

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"testing"

	"github.com/liuy/wukong/assert"
)

func getTokensFrom(path string) (map[string]int32, error) {
	mergeableTokens, err := loadTokenBpe(path)
	if err != nil {
		return nil, err
	}
	mergeableCount := len(mergeableTokens)

	reservedSpecialTokensCount := 256

	tokens := make(map[string]int32, mergeableCount+reservedSpecialTokensCount)
	specialTokensArr := []string{
		"<|begin_of_text|>",
		"<|end_of_text|>",
		"<|reserved_special_token_0|>",
		"<|reserved_special_token_1|>",
		"<|finetune_right_pad_id|>",
		"<|reserved_special_token_2|>",
		"<|start_header_id|>",
		"<|end_header_id|>",
		"<|eom_id|>", // end of message
		"<|eot_id|>", // end of turn
		"<|python_tag|>",
	}

	reservedTokensArr := make([]string, reservedSpecialTokensCount-len(specialTokensArr))
	for i := 0; i < len(reservedTokensArr); i++ {
		reservedTokensArr[i] = fmt.Sprintf("<|reserved_special_token_%d|>", 3+i)
	}
	specialTokensArr = append(specialTokensArr, reservedTokensArr...)

	for id, t := range specialTokensArr {
		tokens[t] = int32(mergeableCount + id)
	}

	for t, id := range mergeableTokens {
		tokens[t] = id
	}
	return tokens, nil
}

func TestTokenizer(t *testing.T) {
	toks, err := getTokensFrom("test_data/llama3_tokenizer.model")
	assert.NoErr(t, err)
	tok := NewTokenizer(toks, &Llama3Handler{})
	expected := 128256
	assert.Equal(t, expected, tok.VocabSize)
	text := "这个世界收到了你们的信息。请不要回答！请不要回答！请不要回答！💖"
	ids := tok.Encode(text)
	expectedIds := []int32{103624, 102616, 51109, 106837, 112022, 9554, 28469,
		1811, 15225, 113473, 113925, 6447, 15225, 113473, 113925,
		6447, 15225, 113473, 113925, 6447, 93273, 244}
	assert.Equal(t, expectedIds, ids)
}

func TestTokenizerEncode(t *testing.T) {
	toks, err := getTokensFrom("test_data/llama3_tokenizer.model")
	assert.NoErr(t, err)
	tok := NewTokenizer(toks, &Llama3Handler{})
	text := "\t\tname"
	ids := tok.Encode(text)
	expectedIds := []int32{197, 11870}
	assert.Equal(t, expectedIds, ids)
	text = "  中国\t 重庆"
	ids = tok.Encode(text)
	expectedIds = []int32{220, 107637, 197, 109367, 110736}
	assert.Equal(t, expectedIds, ids)
	text = "My     name is \twukong. What's your\t name?"
	ids = tok.Encode(text)
	expectedIds = []int32{5159, 257, 836, 374, 220, 6831, 3178, 647, 13, 3639, 596, 701, 197, 836, 30}
	assert.Equal(t, expectedIds, ids)
	text = "   .\t\t\t"
	ids = tok.Encode(text)
	expectedIds = []int32{256, 662, 573}
	assert.Equal(t, expectedIds, ids)
	text = "   123"
	ids = tok.Encode(text)
	expectedIds = []int32{256, 220, 4513}
	assert.Equal(t, expectedIds, ids)
	text = "\t123"
	ids = tok.Encode(text)
	expectedIds = []int32{197, 4513}
	assert.Equal(t, expectedIds, ids)
}

func BenchmarkTokenizer(b *testing.B) {
	toks, err := getTokensFrom("test_data/llama3_tokenizer.model")
	assert.NoErr(b, err)
	tok := NewTokenizer(toks, &Llama3Handler{})
	file, err := os.Open("test_data/shakespeare.txt")
	assert.NoErr(b, err)
	defer file.Close()

	content, err := io.ReadAll(file)
	assert.NoErr(b, err)
	text := string(content)
	ids := tok.Encode(text)
	b.Run("Encode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			tok.Encode(text)
		}
	})
	b.Run("BatchDecode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			tok.BatchDecode(ids)
		}
	})
	assert.Equal(b, text, tok.BatchDecode(ids))
}

func TestGGUFParser(t *testing.T) {
	toks, err := getTokensFrom("test_data/llama3_tokenizer.model")
	assert.NoErr(t, err)
	tok1 := NewTokenizer(toks, &Llama3Handler{})

	gguf, err := GGUFParser("test_data/ggml-vocab-llama-bpe.gguf")
	assert.NoErr(t, err)
	tokens := gguf.GetTokensMap()
	tok2 := NewTokenizer(tokens, &Llama3Handler{})
	for i := 0; i < 12800; i++ {
		assert.Equal(t, tok1.IdToToken[i], tok2.IdToToken[i])
	}
	text := "\"The Matrix is everywhere. It's all around us, even now in this very room. " +
		"You can see it when you look out your window or when you turn on your television. " +
		"You can feel it when you go to work... when you go to church... when you pay your taxes. " +
		"It is the world that has been pulled over your eyes to blind you from the truth.\" - 摩菲斯解释矩阵的本质。"
	ids2 := tok2.Encode(text)
	expected := []int32{10227, 11892, 374, 17277, 13, 1102, 596, 682, 2212, 603, 11, 1524, 1457, 304,
		420, 1633, 3130, 13, 1472, 649, 1518, 433, 994, 499, 1427, 704, 701, 3321, 477, 994, 499, 2543,
		389, 701, 12707, 13, 1472, 649, 2733, 433, 994, 499, 733, 311, 990, 1131, 994, 499, 733, 311,
		8993, 1131, 994, 499, 2343, 701, 13426, 13, 1102, 374, 279, 1917, 430, 706, 1027, 13541, 927,
		701, 6548, 311, 18507, 499, 505, 279, 8206, 1210, 482, 122901, 102, 112789, 101011, 50338, 69962,
		100543, 102, 113400, 9554, 22656, 103706, 1811}
	assert.Equal(t, expected, ids2)
}

func TestGGUFGetPredictor(t *testing.T) {
	m, err := NewModel("test_data/llama3.2-3b.gguf")
	if err != nil {
		t.Skip("llama3.2-3b.gguf: ", err)
	}
	assert.Equal(t, "llama", m.Arch)
	assert.Equal(t, uint32(131072), m.ContextLen)
	assert.Equal(t, uint32(28), m.NumHidden)
	assert.Equal(t, uint32(8192), m.FeedFWDLen)
	assert.Equal(t, uint32(128), m.HeadDim)
	assert.Equal(t, uint32(24), m.NumHead)
	assert.Equal(t, uint32(8), m.NumKVHead)
	assert.Equal(t, uint32(3072), m.EmbedDim)
	assert.Equal(t, float32(1e-05), m.NormEpsilon)
	assert.Equal(t, float32(500000), m.RopeTheta)

	text := "你好，World!"
	ids := m.Encode(text)
	assert.Equal(t, m.BatchDecode(ids), text)

	g := GGUFFile{
		Header: GGUFHeader{},
		KVs: map[string]any{
			"general.architecture":               "llama",
			"llama.context_length":               uint32(131072),
			"llama.block_count":                  uint32(28),
			"llama.attention.head_count":         uint32(24),
			"llama.embedding_length":             uint32(3072),
			"llama.rope.freq_base":               float32(500000),
			"llama.attention.layer_norm_epsilon": float32(1e-05),
			"llama.attention.head_count_kv":      uint32(8),
			"llama.feed_forward_length":          uint32(8192),
		},
	}
	p := g.GetPredictor()
	assert.Equal(t, p.NormEpsilon, m.NormEpsilon)
}

func BenchmarkGGUFParser(b *testing.B) {
	for i := 0; i < b.N; i++ {
		gguf, err := GGUFParser("test_data/ggml-vocab-llama-bpe.gguf")
		assert.NoErr(b, err)
		gguf.GetTokensMap()
	}
}

func TestLoadTensors(t *testing.T) {
	tmpfile, err := os.CreateTemp("", "tensor_test")
	assert.NoErr(t, err)
	defer os.Remove(tmpfile.Name())
	defer tmpfile.Close()

	f32Data := []float32{1.0, 2.0, 3.0, 4.0}
	f64Data := []float64{1.0, 2.0, 3.0, 4.0}
	i8Data := []int8{1, 2, 3, 4}
	i16Data := []int16{1, 2, 3, 4}
	i32Data := []int32{1, 2, 3, 4}
	i64Data := []int64{1, 2, 3, 4}

	f32Offset := uint64(0)
	f64Offset := f32Offset + uint64(binary.Size(f32Data))
	i8Offset := f64Offset + uint64(binary.Size(f64Data))
	i16Offset := i8Offset + uint64(binary.Size(i8Data))
	i32Offset := i16Offset + uint64(binary.Size(i16Data))
	i64Offset := i32Offset + uint64(binary.Size(i32Data))

	writeData := func(data interface{}) error {
		return binary.Write(tmpfile, binary.LittleEndian, data)
	}

	assert.NoErr(t, writeData(f32Data))
	assert.NoErr(t, writeData(f64Data))
	assert.NoErr(t, writeData(i8Data))
	assert.NoErr(t, writeData(i16Data))
	assert.NoErr(t, writeData(i32Data))
	assert.NoErr(t, writeData(i64Data))

	reader, err := MmapOpen(tmpfile.Name())
	assert.NoErr(t, err)
	defer reader.Close()

	gguf := &GGUFFile{
		TensorInfos: map[string]GGUFTensorInfo{
			"f32_tensor": {
				Dims:   []int{2, 2},
				Type:   GGML_TYPE_F32,
				Offset: f32Offset,
			},
			"f64_tensor": {
				Dims:   []int{2, 2},
				Type:   GGML_TYPE_F64,
				Offset: f64Offset,
			},
			"i8_tensor": {
				Dims:   []int{2, 2},
				Type:   GGML_TYPE_I8,
				Offset: i8Offset,
			},
			"i16_tensor": {
				Dims:   []int{2, 2},
				Type:   GGML_TYPE_I16,
				Offset: i16Offset,
			},
			"i32_tensor": {
				Dims:   []int{2, 2},
				Type:   GGML_TYPE_I32,
				Offset: i32Offset,
			},
			"i64_tensor": {
				Dims:   []int{2, 2},
				Type:   GGML_TYPE_I64,
				Offset: i64Offset,
			},
		},
	}

	tensors := loadTensors(reader, gguf)
	assert.NotNil(t, tensors)

	testCases := []struct {
		name  string
		shape Shape
		dtype DType
		data  any
	}{
		{"f32_tensor", Shape{2, 2}, GGML_TYPE_F32, f32Data},
		{"f64_tensor", Shape{2, 2}, GGML_TYPE_F64, f64Data},
		{"i8_tensor", Shape{2, 2}, GGML_TYPE_I8, i8Data},
		{"i16_tensor", Shape{2, 2}, GGML_TYPE_I16, i16Data},
		{"i32_tensor", Shape{2, 2}, GGML_TYPE_I32, i32Data},
		{"i64_tensor", Shape{2, 2}, GGML_TYPE_I64, i64Data},
	}

	for _, tc := range testCases {
		tensor, ok := tensors[tc.name]
		assert.True(t, ok)
		assert.NotNil(t, tensor)
		assert.Equal(t, tc.shape, tensor.Shape)
		assert.Equal(t, tc.dtype, tensor.dtype)
		assert.Equal(t, tc.data, tensor.ToHost())
	}

	badGGUF := &GGUFFile{
		TensorInfos: map[string]GGUFTensorInfo{
			"bad_tensor": {
				Dims:   []int{2, 2},
				Type:   GGML_TYPE_F32,
				Offset: 999999999, // Invalid offset
			},
		},
	}
	tensors = loadTensors(reader, badGGUF)
	assert.Nil(t, tensors)

	badGGUF = &GGUFFile{
		TensorInfos: map[string]GGUFTensorInfo{
			"bad_tensor": {
				Dims:   []int{0}, // Invalid dimension
				Type:   GGML_TYPE_F32,
				Offset: 0,
			},
		},
	}
	tensors = loadTensors(reader, badGGUF)
	assert.Nil(t, tensors)
}

func TestMmapReader(t *testing.T) {
	content := []byte("Hello, World! This is a test file.")
	tmpfile, err := os.CreateTemp("", "mmaptest")
	assert.NoErr(t, err)
	defer os.Remove(tmpfile.Name())

	_, err = tmpfile.Write(content)
	assert.NoErr(t, err)
	err = tmpfile.Close()
	assert.NoErr(t, err)

	t.Run("Open", func(t *testing.T) {
		reader, err := MmapOpen(tmpfile.Name())
		assert.NoErr(t, err)
		defer reader.Close()

		assert.Equal(t, len(content), reader.Len())

		zeroSizeFile, err := os.CreateTemp("", "mmaptest_zero_size")
		assert.NoErr(t, err)
		defer os.Remove(zeroSizeFile.Name())
		err = zeroSizeFile.Close()
		assert.NoErr(t, err)
		_, err = MmapOpen(zeroSizeFile.Name())
		assert.Error(t, err)
	})

	t.Run("Read", func(t *testing.T) {
		reader, err := MmapOpen(tmpfile.Name())
		assert.NoErr(t, err)
		defer reader.Close()

		buf := make([]byte, 5)
		n, err := reader.Read(buf)
		assert.NoErr(t, err)
		assert.Equal(t, 5, n)
		assert.Equal(t, "Hello", string(buf))
	})

	t.Run("ReadAt", func(t *testing.T) {
		reader, err := MmapOpen(tmpfile.Name())
		assert.NoErr(t, err)
		defer reader.Close()

		buf := make([]byte, 5)
		n, err := reader.ReadAt(buf, 7)
		assert.NoErr(t, err)
		assert.Equal(t, 5, n)
		assert.Equal(t, "World", string(buf))
	})

	t.Run("AlignOffset", func(t *testing.T) {
		reader, err := MmapOpen(tmpfile.Name())
		assert.NoErr(t, err)
		defer reader.Close()

		reader.offset = 5
		aligned := reader.AlignOffset(8)
		assert.Equal(t, int64(8), aligned)
	})

	t.Run("PointerAt", func(t *testing.T) {
		reader, err := MmapOpen(tmpfile.Name())
		assert.NoErr(t, err)
		defer reader.Close()

		ptr, err := reader.PointerAt(0)
		assert.NoErr(t, err)
		assert.NotNil(t, ptr)
		b := *(*byte)(ptr)
		assert.Equal(t, content[0], b)

		ptr, err = reader.PointerAt(7)
		assert.NoErr(t, err)
		assert.NotNil(t, ptr)
		b = *(*byte)(ptr)
		assert.Equal(t, content[7], b)
	})

	t.Run("ErrorCases", func(t *testing.T) {
		// Test reading from closed reader
		reader, err := MmapOpen(tmpfile.Name())
		assert.NoErr(t, err)
		reader.Close()

		buf := make([]byte, 5)
		_, err = reader.Read(buf)
		assert.Error(t, err)

		// Test invalid ReadAt offset
		reader, err = MmapOpen(tmpfile.Name())
		assert.NoErr(t, err)

		_, err = reader.ReadAt(buf, -1)
		assert.Error(t, err)

		_, err = reader.ReadAt(buf, int64(len(content)+1))
		assert.Error(t, err)

		_, err = reader.PointerAt(-1)
		assert.Error(t, err)

		_, err = reader.PointerAt(int64(len(content) + 1))
		assert.Error(t, err)

		reader.Close()
		_, err = reader.PointerAt(0)
		assert.Error(t, err)
	})

	t.Run("EOF", func(t *testing.T) {
		reader, err := MmapOpen(tmpfile.Name())
		assert.NoErr(t, err)
		defer reader.Close()

		buf := make([]byte, 100)
		n, err := reader.Read(buf)
		assert.Equal(t, io.EOF, err)
		assert.Equal(t, len(content), n)
	})

	t.Run("NonExistentFile", func(t *testing.T) {
		_, err := MmapOpen("nonexistent.file")
		assert.Error(t, err)
	})
}

func TestModelGenerate(t *testing.T) {
	m, err := NewModel("test_data/llama3.2-1b-f32.gguf")
	if err != nil {
		t.Skip("llama3.2-3b.gguf: ", err)
	}
	m.Setup()
	message := map[string]string{
		"system": "You are a helpful assistant",
		"user":   "What is the capital of China?",
	}
	m.Generate(message)
	fmt.Println()
}
