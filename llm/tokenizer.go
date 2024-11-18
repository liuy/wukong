package llm

import (
	"bufio"
	"encoding/base64"
	"log/slog"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
)

type ModelHandler interface {
	Initialize(t *Tokenizer, toks map[string]int)
	EncodeHeader(t *Tokenizer, message map[string]string) []int
	EncodeContent(t *Tokenizer, message map[string]string) []int
}

type Tokenizer struct {
	TokenToId map[string]int
	IdToToken []string
	Pattern   *regexp.Regexp
	VocabSize int
	BosId     int
	EosId     int
	EomId     int
	EotId     int
	PadId     int
	UnknownId int
	ModelHandler
}

func NewTokenizer(tokens map[string]int, m ModelHandler) *Tokenizer {
	tokenizer := &Tokenizer{}
	tokenizer.ModelHandler = m
	tokenizer.Initialize(tokens)
	return tokenizer
}

func loadTokenBpe(vocabFilePath string) (map[string]int, error) {
	vocabFile, err := os.Open(vocabFilePath)
	if err != nil {
		return nil, err
	}
	defer vocabFile.Close()

	fileScanner := bufio.NewScanner(vocabFile)
	fileScanner.Split(bufio.ScanLines)

	result := make(map[string]int)

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
		result[string(token)] = id
	}
	return result, nil
}

func (t *Tokenizer) BytePairMerge(piece string) []int {
	parts := make([]struct {
		rank int
		idx  int
	}, len(piece)+1)

	minRank := math.MaxInt32
	minIdx := 0
	strLen := len(piece)

	// Initial ranking pass
	for i := 0; i < strLen-1; i++ {
		rank := math.MaxInt32

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

	getRankFn := func(parts []struct{ rank, idx int }, i int) int {
		if i+3 >= len(parts) {
			return math.MaxInt32
		}

		pieceToSearch := piece[parts[i].idx:parts[i+3].idx]
		if rank, ok := t.TokenToId[pieceToSearch]; ok {
			return rank
		}
		return math.MaxInt32
	}

	// Main merge loop
	partsLen := len(piece) + 1
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
		for j := 0; j < partsLen-1; j++ {
			if parts[j].rank < minRank {
				minRank = parts[j].rank
				minIdx = j
			}
		}
	}

	// Build result
	splitIds := make([]int, 0, partsLen-1)
	for i := 0; i < partsLen-1; i++ {
		start := parts[i].idx
		end := parts[i+1].idx
		token := piece[start:end]
		id, ok := t.TokenToId[token]
		if !ok {
			id = t.UnknownId
			slog.Warn("Tokenizer found unknown token: " + token)
		}
		splitIds = append(splitIds, id)
	}

	return splitIds
}

// Convert text to a sequence of token IDs
func (t *Tokenizer) Encode(text string) []int {
	estimatedTokens := len(text) / 4
	ids := make([]int, 0, estimatedTokens)

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
func (t *Tokenizer) Decode(ids []int) string {
	// Assuming average token length of 4 characters as an estimation
	capacity := len(ids) * 4

	var sb strings.Builder
	sb.Grow(capacity)

	for _, id := range ids {
		sb.WriteString(t.IdToToken[id])
	}

	return sb.String()
}

func (t *Tokenizer) EncodeHeader(message map[string]string) []int {
	return t.ModelHandler.EncodeHeader(t, message)
}

func (t *Tokenizer) EncodeContent(message map[string]string) []int {
	return t.ModelHandler.EncodeContent(t, message)
}

func (t *Tokenizer) Initialize(toks map[string]int) {
	t.ModelHandler.Initialize(t, toks)
}
