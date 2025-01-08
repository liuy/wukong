package llm

import (
	"bufio"
	"encoding/base64"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
)

type TokenHandler interface {
	Initialize(t *Tokenizer, toks map[string]int32)
	EncodeHeader(t *Tokenizer, message string) []int32
	EncodeMessage(t *Tokenizer, message map[string]string) []int32
}

type Tokenizer struct {
	TokenToId map[string]int32
	IdToToken []string
	Pattern   *regexp.Regexp
	VocabSize int
	BosId     int32
	EosId     int32
	EomId     int32
	EotId     int32
	PadId     int32
	UnknownId int32
	TokenHandler
}

func NewTokenizer(tokens map[string]int32, m TokenHandler) *Tokenizer {
	tokenizer := &Tokenizer{}
	tokenizer.TokenHandler = m
	tokenizer.Initialize(tokens)
	return tokenizer
}

func loadTokenBpe(vocabFilePath string) (map[string]int32, error) {
	vocabFile, err := os.Open(vocabFilePath)
	if err != nil {
		return nil, err
	}
	defer vocabFile.Close()

	fileScanner := bufio.NewScanner(vocabFile)
	fileScanner.Split(bufio.ScanLines)

	result := make(map[string]int32)

	for fileScanner.Scan() {
		lineParts := strings.Split(fileScanner.Text(), " ")
		token, err := base64.StdEncoding.DecodeString(lineParts[0])
		if err != nil {
			return nil, err
		}
		id, err := strconv.Atoi(lineParts[1])
		if err != nil {
			return nil, err
		}
		result[string(token)] = int32(id)
	}
	return result, nil
}

func (t *Tokenizer) BytePairMerge(piece string) []int32 {
	parts := make([]struct {
		rank int32
		idx  int32
	}, len(piece)+1)

	var minRank int32 = math.MaxInt32
	var minIdx int32 = 0
	var strLen int32 = int32(len(piece))

	// Initial ranking pass
	for i := int32(0); i < strLen-1; i++ {
		var rank int32 = math.MaxInt32

		// Check if we have a valid 2-byte sequence
		if i+1 < strLen {
			key := piece[i : i+2]
			if r, ok := t.TokenToId[key]; ok {
				rank = r
			}
		}

		if rank < minRank {
			minRank = rank
			minIdx = i
		}
		parts[i].rank = rank
		parts[i].idx = i
	}

	// Set sentinel values
	parts[strLen-1].rank = math.MaxInt32
	parts[strLen-1].idx = strLen - 1
	parts[strLen].rank = math.MaxInt32
	parts[strLen].idx = strLen

	getRankFn := func(parts []struct{ rank, idx int32 }, i int32) int32 {
		if i+3 >= int32(len(parts)) {
			return math.MaxInt32
		}

		pieceToSearch := piece[parts[i].idx:parts[i+3].idx]
		if rank, ok := t.TokenToId[pieceToSearch]; ok {
			return rank
		}
		return math.MaxInt32
	}

	// Main merge loop
	partsLen := int32(len(piece) + 1)
	for minRank != math.MaxInt32 {
		i := minIdx
		// Update ranks
		// Update parts[i] and parts[i - 1] before removing parts[i + 1], since
		// `parts.remove(i + 1)` will thrash the cache.
		if i > 0 {
			parts[i-1].rank = getRankFn(parts[:partsLen], i-1)
		}
		parts[i].rank = getRankFn(parts[:partsLen], i)

		// Remove parts[i + 1]
		copy(parts[i+1:partsLen-1], parts[i+2:partsLen])
		partsLen--

		// Find new minimum
		minRank = math.MaxInt32
		for j := int32(0); j < partsLen-1; j++ {
			if parts[j].rank < minRank {
				minRank = parts[j].rank
				minIdx = j
			}
		}
	}

	// Build result
	splitIds := make([]int32, 0, partsLen-1)
	for i := int32(0); i < partsLen-1; i++ {
		start := parts[i].idx
		end := parts[i+1].idx
		token := piece[start:end]
		id, ok := t.TokenToId[token]
		if !ok {
			id = t.UnknownId
		}
		splitIds = append(splitIds, id)
	}

	return splitIds
}

// Convert text to a sequence of token IDs
func (t *Tokenizer) Encode(text string) []int32 {
	estimatedTokens := len(text) / 4
	ids := make([]int32, 0, estimatedTokens)

	matches := t.Pattern.FindAllStringSubmatchIndex(text, -1)
	for i, match := range matches {
		isSpaceOrTab := func(b byte) bool {
			return b == ' ' || b == '\t'
		}
		// mimic lookahead \s+(?!\S) since Go's regexp doesn't support it
		if isSpaceOrTab(text[match[1]-1]) && match[1]-match[0] > 1 && i+1 < len(matches) {
			next := matches[i+1]
			// move the tail whitespace of current match to the next match
			next[0] -= 1
			match[1] -= 1
		}
		start, end := match[0], match[1]
		piece := text[start:end]

		if id, ok := t.TokenToId[piece]; ok { // First check if the piece is in the vocabulary
			ids = append(ids, id)
		} else { // Otherwise, split the piece into subtokens by BPE
			splitIds := t.BytePairMerge(piece)
			ids = append(ids, splitIds...)
		}
	}

	return ids
}

// Convert a sequence of token IDs to text
func (t *Tokenizer) BatchDecode(ids []int32) string {
	// Assuming average token length of 4 characters as an estimation
	capacity := len(ids) * 4

	var sb strings.Builder
	sb.Grow(capacity)

	for _, id := range ids {
		sb.WriteString(t.IdToToken[id])
	}

	return sb.String()
}

func (t *Tokenizer) Decode(id int32) string {
	return t.IdToToken[id]
}

func (t *Tokenizer) EncodeHeader(role string) []int32 {
	return t.TokenHandler.EncodeHeader(t, role)
}

func (t *Tokenizer) EncodeMessage(message map[string]string) []int32 {
	return t.TokenHandler.EncodeMessage(t, message)
}

func (t *Tokenizer) Initialize(toks map[string]int32) {
	t.TokenHandler.Initialize(t, toks)
}
