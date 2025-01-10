package llm

import (
	"fmt"
	"os"
	"os/signal"

	"github.com/liuy/wukong/cmd"
)

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

var pbar = cmd.NewProgressBar()

func NewModel(path string) (*Model, error) {
	pbar.PrefixText = "Awakening "
	pbar.ShowPercentage = true
	cmd.HideCursor()
	gguf, err := GGUFParser(path)
	if err != nil {
		return nil, err
	}
	Predictor := gguf.GetPredictor()
	Tokenizer := gguf.GetTokenizer()
	return &Model{Predictor, Tokenizer}, nil
}

func (m *Model) Generate(message map[string]string) error {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt)
	defer signal.Stop(sigChan)

	s := m.EncodeMessage(message)
	var ids [][]int32
	ids = append(ids, s)
	for {
		select {
		case <-sigChan:
			fmt.Print("[Cancelled by Ctrl+C]")
			return nil
		default:
			pids, err := m.Predict(ids)
			if err != nil {
				return err
			}
			if pids[0] == m.EotId || pids[0] == -1 {
				return nil
			}
			ids[0] = append(ids[0], pids[0])
			fmt.Print(m.Decode(pids[0]))
		}
	}
}

func (m *Model) Setup() error {
	e := m.PredicHandler.Setup(m.Predictor)
	pbar.Tick()
	cmd.ShowCursor()
	return e
}

func (m *Model) Predict(ids [][]int32) ([]int32, error) {
	return m.PredicHandler.Predict(m.Predictor, ids)
}
