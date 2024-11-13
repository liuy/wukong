package llm

import (
	"io"
	"os"
	"slices"
	"testing"
)

func TestTokenizer(t *testing.T) {
	tok, err := NewTokenizer("./llama3_tokenizer.model", &Llama3Handler{})
	if err != nil {
		t.Error(err)
	}
	expected := 128256
	if tok.VocabSize != expected {
		t.Errorf("Tokenizer.VocabSize = %d, want %d", tok.VocabSize, expected)
	}
	text := "这个世界收到了你们的信息。请不要回答！请不要回答！请不要回答！💖"
	ids := tok.Encode(text)
	expectedIds := []int{103624, 102616, 51109, 106837, 112022, 9554, 28469,
		1811, 15225, 113473, 113925, 6447, 15225, 113473, 113925,
		6447, 15225, 113473, 113925, 6447, 93273, 244}
	if !slices.Equal(expectedIds, ids) {
		t.Errorf("Tokenizer.Encode() = %v, want %v", ids, expectedIds)
	}
}

func BenchmarkTokenizer(b *testing.B) {
	tok, err := NewTokenizer("./llama3_tokenizer.model", &Llama3Handler{})
	if err != nil {
		b.Fatalf("NewTokenizer() error = %v", err)
	}
	file, err := os.Open("shakespeare.txt")
	if err != nil {
		b.Fatalf("os.Open() error = %v", err)
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		b.Fatalf("io.ReadAll() error = %v", err)
	}
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
	if text != tok.Decode(ids) {
		b.Fatalf("text != tok.Decode(ids)")
	}
}

func TestGGUFParser(t *testing.T) {
	gguf, err := GGUFParser("test_data/ggml-vocab-gpt-2.gguf")
	if err != nil {
		t.Fatalf("GGUFParser() error = %v", err)
	}
	if gguf.Header.KVCount != 16 || gguf.Header.TensorCount != 0 {
		t.Errorf("GGUFFile.Header = %v, want KVCount = 16, TensorCount = 0", gguf.Header)
	}
	if gguf.KVs["gpt2.context_length"].(uint32) != 1024 {
		t.Errorf("Got gtp2.context_length:%d, want 1024", gguf.KVs["gpt2.context_length"])
	}
	if len(gguf.KVs["tokenizer.ggml.tokens"].([]string)) != 50257 {
		t.Errorf("Got len(tokenizer.ggml.tokens):%d, want 50257", len(gguf.KVs["tokenizer.ggml.tokens"].([]string)))
	}
}
