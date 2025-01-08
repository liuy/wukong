package llm

import "fmt"

type PredicHandler interface {
	Setup(*Predictor) error
	Predict(*Predictor, [][]int32) ([]int32, error)
}

type Predictor struct {
	Arch        string  // "llama"
	ContextLen  uint32  // context length
	NumHidden   uint32  // number of hidden layers
	FeedFWDLen  uint32  // feed forward length
	HeadDim     uint32  // head dimension
	NumHead     uint32  // number of attention heads
	NumKVHead   uint32  // number of key and value heads
	EmbedDim    uint32  // word token embedding dimension
	NormEpsilon float32 // normalization epsilon
	RopeTheta   float32 // rope theta value
	Tensors     map[string]*Tensor
	PredicHandler
}

type Model struct {
	*Predictor
	*Tokenizer
}

func NewModel(path string) (*Model, error) {
	gguf, err := GGUFParser(path)
	if err != nil {
		return nil, err
	}
	return &Model{
		Predictor: gguf.GetPredictor(),
		Tokenizer: gguf.GetTokenizer(),
	}, nil
}

func (m *Model) Generate(message map[string]string) error {
	s := m.EncodeMessage(message)
	var ids [][]int32
	ids = append(ids, s)
	for {
		pids, err := m.Predict(ids)
		if err != nil {
			return err
		}
		if pids[0] == m.EotId || pids[0] == -1 {
			break
		}
		ids[0] = append(ids[0], pids[0])
		fmt.Print(m.Decode(pids[0]))
	}
	return nil
}

func (m *Model) Setup() error {
	return m.PredicHandler.Setup(m.Predictor)
}

func (m *Model) Predict(ids [][]int32) ([]int32, error) {
	return m.PredicHandler.Predict(m.Predictor, ids)
}
