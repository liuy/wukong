package llm

import (
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

type Llama3Handler struct{}

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

func (m *Llama3Handler) Initialize(toker *Tokenizer, toks map[string]int32) {
	toker.TokenToId = make(map[string]int32, len(toks))
	toker.IdToToken = make([]string, len(toks))
	for token, id := range toks {
		toker.TokenToId[token] = id
		toker.IdToToken[id] = token
	}
	toker.VocabSize = len(toks)
	toker.BosId = toks["<|begin_of_text|>"]
	toker.EosId = toks["<|end_of_text|>"]
	toker.EomId = toks["<|eom_id|>"]
	toker.EotId = toks["<|eot_id|>"]
	toker.PadId = -1
	toker.UnknownId = -1
	toker.Pattern = regexp.MustCompile(`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`)
}

// <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\n
func (m *Llama3Handler) EncodeHeader(t *Tokenizer, role string) []int32 {
	ids := []int32{}
	ids = append(ids, t.TokenToId["<|start_header_id|>"])
	ids = append(ids, t.Encode(role)...)
	ids = append(ids, t.TokenToId["<|end_header_id|>"])
	ids = append(ids, t.Encode("\n\n")...)
	return ids
}

func (m *Llama3Handler) EncodeMessage(t *Tokenizer, message map[string]string) []int32 {
	var ids []int32
	ids = append(ids, t.BosId)                                           // <|begin_of_text|>
	ids = append(ids, t.EncodeHeader("system")...)                       // <|start_header_id|>system<|end_header_id|>
	ids = append(ids, t.Encode(strings.TrimSpace(message["system"]))...) // system content...
	ids = append(ids, t.EotId)                                           // <|eot_id|>
	ids = append(ids, t.EncodeHeader("user")...)                         // <|start_header_id|>user<|end_header_id|>
	ids = append(ids, t.Encode(strings.TrimSpace(message["user"]))...)   // user content...
	ids = append(ids, t.EotId)                                           // <|eot_id|>
	ids = append(ids, t.EncodeHeader("assistant")...)                    // <|start_header_id|>assistant<|end_header_id|>

	return ids
}

func get_freqs_array(HS int, theta float32) *Tensor {
	freqs := make([]float32, HS/2)
	for i := 0; i < HS/2; i++ {
		freq := 1.0 / float32(math.Pow(float64(theta), float64(2*i)/float64(HS)))
		freqs[i] = freq
	}
	f, err := MakeTensor(Shape{HS / 2}, freqs)
	if err != nil {
		panic(err)
	}
	return f
}

func (m *Llama3Handler) Setup(pred *Predictor) error {
	w := pred.Tensors["rope_freqs.weight"]
	freqs := get_freqs_array(int(pred.HeadDim), pred.RopeTheta)
	freqs.DivInPlace(w)
	pred.Tensors["rope_freqs.weight"] = freqs
	w.DeviceFree()

	output := pred.Tensors["output.weight"]
	if output == nil {
		pred.Tensors["output.weight"] = pred.Tensors["token_embd.weight"]
	}

	for i := uint32(0); i < pred.NumHidden; i++ {
		q := pred.Tensors["blk."+strconv.Itoa(int(i))+".attn_q.weight"]
		k := pred.Tensors["blk."+strconv.Itoa(int(i))+".attn_k.weight"]
		v := pred.Tensors["blk."+strconv.Itoa(int(i))+".attn_v.weight"]
		qk, err := q.Cat(k)
		if err != nil {
			return err
		}
		qkv, err := qk.Cat(v)
		if err != nil {
			return err
		}
		pred.Tensors["blk."+strconv.Itoa(int(i))+".attn_qkv.weight"] = qkv
		delete(pred.Tensors, "blk."+strconv.Itoa(int(i))+".attn_q.weight")
		delete(pred.Tensors, "blk."+strconv.Itoa(int(i))+".attn_k.weight")
		delete(pred.Tensors, "blk."+strconv.Itoa(int(i))+".attn_v.weight")
		q.DeviceFree()
		k.DeviceFree()
		v.DeviceFree()
		qk.DeviceFree()

		up := pred.Tensors["blk."+strconv.Itoa(int(i))+".ffn_up.weight"]
		gate := pred.Tensors["blk."+strconv.Itoa(int(i))+".ffn_gate.weight"]
		fc, err := up.Cat(gate)
		if err != nil {
			return err
		}
		pred.Tensors["blk."+strconv.Itoa(int(i))+".ffn_fc.weight"] = fc
		delete(pred.Tensors, "blk."+strconv.Itoa(int(i))+".ffn_up.weight")
		delete(pred.Tensors, "blk."+strconv.Itoa(int(i))+".ffn_gate.weight")
		up.DeviceFree()
		gate.DeviceFree()
	}

	return nil
}

func (m *Llama3Handler) Predict(pred *Predictor, toks [][]int32) ([]int32, error) {
	t := pred.Tensors["token_embd.weight"]
	wcsize := t.GetDim(0)
	batch := len(toks)
	seqs := len(toks[0])
	flat := slices.Concat(toks...)
	ids, err := MakeTensor(Shape{batch, seqs}, flat)
	if err != nil {
		return nil, err
	}
	embeds, err := t.Embedding(ids)
	if err != nil {
		return nil, err
	}
	freqs := pred.Tensors["rope_freqs.weight"]
	for i := uint32(0); i < pred.NumHidden; i++ {
		qkv := pred.Tensors["blk."+strconv.Itoa(int(i))+".attn_qkv.weight"]
		attn_norm := pred.Tensors["blk."+strconv.Itoa(int(i))+".attn_norm.weight"]
		attn_output := pred.Tensors["blk."+strconv.Itoa(int(i))+".attn_output.weight"]
		ffn_norm := pred.Tensors["blk."+strconv.Itoa(int(i))+".ffn_norm.weight"]
		ffn_fc := pred.Tensors["blk."+strconv.Itoa(int(i))+".ffn_fc.weight"]
		ffn_down := pred.Tensors["blk."+strconv.Itoa(int(i))+".ffn_down.weight"]

		err := embeds.GroupQueryAttention(freqs, attn_norm, qkv, attn_output, int(pred.NumHead), int(pred.NumKVHead), pred.NormEpsilon)
		if err != nil {
			panic(err)
		}
		err = embeds.FeedForward(ffn_norm, ffn_fc, ffn_down, int(pred.FeedFWDLen), pred.NormEpsilon)
		if err != nil {
			panic(err)
		}
	}
	out_norm := pred.Tensors["output_norm.weight"]
	out_weight := pred.Tensors["output.weight"]
	tids := embeds.Predict(out_norm, out_weight, wcsize, pred.NormEpsilon)

	embeds.DeviceFree()
	ids.DeviceFree()
	return tids, nil
}
