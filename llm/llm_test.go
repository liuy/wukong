package llm

import (
	"io"
	"os"
	"testing"

	"github.com/liuy/wukong/assert"
)

func TestTokenizer(t *testing.T) {
	toks, err := getTokensFrom("test_data/llama3_tokenizer.model")
	assert.NoErr(t, err)
	tok := NewTokenizer(toks, &Llama3Handler{})
	expected := 128256
	assert.Equal(t, expected, tok.VocabSize)
	text := "这个世界收到了你们的信息。请不要回答！请不要回答！请不要回答！💖"
	ids := tok.Encode(text)
	expectedIds := []int{103624, 102616, 51109, 106837, 112022, 9554, 28469,
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
	expectedIds := []int{197, 11870}
	assert.Equal(t, expectedIds, ids)
	text = "  中国\t 重庆"
	ids = tok.Encode(text)
	expectedIds = []int{220, 107637, 197, 109367, 110736}
	assert.Equal(t, expectedIds, ids)
	text = "My     name is \twukong. What's your\t name?"
	ids = tok.Encode(text)
	expectedIds = []int{5159, 257, 836, 374, 220, 6831, 3178, 647, 13, 3639, 596, 701, 197, 836, 30}
	assert.Equal(t, expectedIds, ids)
	text = "   .\t\t\t"
	ids = tok.Encode(text)
	expectedIds = []int{256, 662, 573}
	assert.Equal(t, expectedIds, ids)
	text = "   123"
	ids = tok.Encode(text)
	expectedIds = []int{256, 220, 4513}
	assert.Equal(t, expectedIds, ids)
	text = "\t123"
	ids = tok.Encode(text)
	expectedIds = []int{197, 4513}
	assert.Equal(t, expectedIds, ids)
}

func BenchmarkTokenizer(b *testing.B) {
	toks, err := getTokensFrom("test_data/llama3_tokenizer.model")
	assert.NoErr(b, err)
	tok := NewTokenizer(toks, &Llama3Handler{})
	file, err := os.Open("shakespeare.txt")
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
	b.Run("Decode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			tok.Decode(ids)
		}
	})
	assert.Equal(b, text, tok.Decode(ids))
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
	expected := []int{10227, 11892, 374, 17277, 13, 1102, 596, 682, 2212, 603, 11, 1524, 1457, 304,
		420, 1633, 3130, 13, 1472, 649, 1518, 433, 994, 499, 1427, 704, 701, 3321, 477, 994, 499, 2543,
		389, 701, 12707, 13, 1472, 649, 2733, 433, 994, 499, 733, 311, 990, 1131, 994, 499, 733, 311,
		8993, 1131, 994, 499, 2343, 701, 13426, 13, 1102, 374, 279, 1917, 430, 706, 1027, 13541, 927,
		701, 6548, 311, 18507, 499, 505, 279, 8206, 1210, 482, 122901, 102, 112789, 101011, 50338, 69962,
		100543, 102, 113400, 9554, 22656, 103706, 1811}
	assert.Equal(t, expected, ids2)
}

func BenchmarkGGUFParser(b *testing.B) {
	for i := 0; i < b.N; i++ {
		gguf, err := GGUFParser("test_data/ggml-vocab-llama-bpe.gguf")
		assert.NoErr(b, err)
		gguf.GetTokensMap()
	}
}
