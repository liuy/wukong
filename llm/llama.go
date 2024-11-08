package llm

import (
	"fmt"
	"regexp"
	"strings"
)

type Llama3Handler struct{}

func (m *Llama3Handler) Initialize(toker *Tokenizer, path string) error {
	mergeableTokens, err := loadTokenBpe(path)
	if err != nil {
		return err
	}
	mergeableCount := len(mergeableTokens)

	reservedSpecialTokensCount := 256

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

	toker.TokenToId = make(map[string]int)
	toker.IdToToken = make([]string, mergeableCount+reservedSpecialTokensCount)

	specialTokens := make(map[string]int)
	for i, token := range specialTokensArr {
		specialTokens[token] = mergeableCount + i
	}

	for token, id := range mergeableTokens {
		toker.TokenToId[token] = id
	}
	for token, id := range specialTokens {
		toker.TokenToId[token] = id
	}
	for token, id := range mergeableTokens {
		toker.IdToToken[id] = token
	}
	for token, id := range specialTokens {
		toker.IdToToken[id] = token
	}

	toker.VocabSize = mergeableCount + reservedSpecialTokensCount
	toker.BosId = specialTokens["<|begin_of_text|>"]
	toker.EosId = specialTokens["<|end_of_text|>"]
	toker.EomId = specialTokens["<|eom_id|>"]
	toker.EotId = specialTokens["<|eot_id|>"]
	toker.PadId = -1
	toker.UnknownId = -1
	toker.Pattern = regexp.MustCompile(`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`)
	return nil
}

// <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\n
func (m *Llama3Handler) EncodeHeader(t *Tokenizer, message map[string]string) []int {
	ids := []int{}
	ids = append(ids, t.TokenToId["<|start_header_id|>"])
	ids = append(ids, t.Encode(message["role"])...)
	ids = append(ids, t.TokenToId["<|end_header_id|>"])
	ids = append(ids, t.Encode("\n\n")...)
	return ids
}

// content<|eot_id|>
func (m *Llama3Handler) EncodeContent(t *Tokenizer, message map[string]string) []int {
	ids := t.EncodeHeader(message)
	ids = append(ids, t.Encode(strings.TrimSpace(message["content"]))...)
	ids = append(ids, t.EotId)
	return ids
}
