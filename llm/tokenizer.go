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

type ModelHandler interface {
	Initialize(t *Tokenizer, path string) error
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

func NewTokenizer(path string, m ModelHandler) (*Tokenizer, error) {
	tokenizer := &Tokenizer{}
	tokenizer.ModelHandler = m
	err := tokenizer.Initialize(path)
	if err != nil {
		return nil, err
	}
	return tokenizer, nil
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
	// Ported from Tiktoken Rust code
	// See: https://github.com/openai/tiktoken/blob/1b9faf2779855124f05174adf1383e53689ed94b/src/lib.rs
	type rankTuple struct {
		rank int
		idx  int
	}
	parts := make([]rankTuple, len(piece)+1)
	min_rank := rankTuple{rank: math.MaxInt32, idx: math.MaxInt32}
	for i := 0; i < len(piece)-1; i++ {
		var rank int
		var ok bool
		if i+1 < len(piece) {
			if rank, ok = t.TokenToId[piece[i:i+2]]; !ok {
				rank = math.MaxInt32
			}
		} else {
			rank = math.MaxInt32
		}
		if rank < min_rank.rank {
			min_rank = rankTuple{rank: rank, idx: i}
		}
		parts[i] = rankTuple{rank: rank, idx: i}
	}
	parts[len(piece)-1] = rankTuple{rank: math.MaxInt32, idx: len(piece) - 1}
	parts[len(piece)] = rankTuple{rank: math.MaxInt32, idx: len(piece)}

	getRankFn := func(parts []rankTuple, i int) int {
		var newRank int
		var ok bool
		if i+3 < len(parts) {
			pieceToSearch := piece[parts[i].idx:parts[i+3].idx]
			if newRank, ok = t.TokenToId[pieceToSearch]; !ok {
				newRank = math.MaxInt32
			}
		} else {
			newRank = math.MaxInt32
		}
		return newRank
	}

	for min_rank.rank != math.MaxInt32 {
		i := min_rank.idx
		// Update parts[i] and parts[i - 1] before removing parts[i + 1], since
		// `parts.remove(i + 1)` will thrash the cache.
		if i > 0 {
			parts[i-1].rank = getRankFn(parts, i-1)
		}
		parts[i].rank = getRankFn(parts, i)
		parts = append(parts[:i+1], parts[i+1+1:]...) // remove parts[i + 1]

		min_rank = rankTuple{rank: math.MaxInt32, idx: math.MaxInt}
		for i = 0; i < len(parts)-1; i++ {
			if parts[i].rank < min_rank.rank {
				min_rank = rankTuple{rank: parts[i].rank, idx: i}
			}
		}
	}

	splitRanks := make([]int, 0)
	for i := 0; i < len(parts)-1; i++ {
		subPiece := piece[parts[i].idx:parts[i+1].idx]
		splitRanks = append(splitRanks, t.TokenToId[subPiece])
	}
	return splitRanks
}

func (t *Tokenizer) Encode(text string) []int {
	ids := make([]int, 0)

	for _, match := range t.Pattern.FindAllStringSubmatch(text, -1) {
		if id, ok := t.TokenToId[match[0]]; ok {
			ids = append(ids, id)
			continue
		}
		splitIds := t.BytePairMerge(match[0])
		ids = append(ids, splitIds...)
	}
	return ids
}

func (t *Tokenizer) Decode(ids []int) string {
	text := ""
	for _, id := range ids {
		text += t.IdToToken[id]
	}
	return text
}

func (t *Tokenizer) EncodeHeader(message map[string]string) []int {
	return t.ModelHandler.EncodeHeader(t, message)
}

func (t *Tokenizer) EncodeContent(message map[string]string) []int {
	return t.ModelHandler.EncodeContent(t, message)
}

func (t *Tokenizer) Initialize(path string) error {
	return t.ModelHandler.Initialize(t, path)
}
