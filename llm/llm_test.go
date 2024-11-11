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
	// open the shakespeare.txt file and assign the content to the variable text
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
