package llm

import (
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
