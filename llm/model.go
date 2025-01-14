package llm

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/liuy/wukong/cmd"
)

var Version = "0.1"

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

type registryOptions struct {
	Insecure bool
	Username string
	Password string
	Token    string

	CheckRedirect func(req *http.Request, via []*http.Request) error
}

type Model struct {
	*Predictor
	*Tokenizer
}

var pbar = NewProgressBar()

func PullModel(ctx context.Context, path string) (string, error) {
	regOpts := &registryOptions{
		Insecure: true,
	}

	mp := ParseModelPath(path)
	prune, err := BuildPruneMap(mp)
	if err != nil {
		return "", err
	}

	manifest, err := PullManifest(ctx, mp, regOpts)
	if err != nil {
		return "", err
	}

	err = PullLayers(ctx, mp, regOpts, manifest, prune)
	if err != nil {
		return "", err
	}

	if err := WriteManifest(mp, manifest); err != nil {
		return "", err
	}

	if err := PruneLayers(prune); err != nil {
		return "", err
	}

	if err := LinkModel(mp, manifest); err != nil {
		return "", err
	}

	return mp.GetModelTagPath(), nil
}

func NewModel(ctx context.Context, path string) (*Model, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) { // First check if the model path exists
		// If not, check if the model is in .wukong/models
		mp := ParseModelPath(path)
		p := mp.GetModelTagPath()
		if _, err := os.Stat(p); os.IsNotExist(err) {
			// Finally, pull the model from the internet
			if path, err = PullModel(ctx, path); err != nil {
				return nil, err
			}
		} else {
			path = p
		}
	}
	pbar.Reset()
	pbar.PrefixText = "AWAKENING "
	pbar.ShowPercentage = true
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
	buf, err := cmd.NewBuffer(&cmd.Prompt{})
	if err != nil {
		return err
	}
	part := strings.Builder{}
	start := time.Now()
	numtok := 0
	for {
		select {
		case <-sigChan:
			fmt.Println("[Cancelled by Ctrl+C]")
			return nil
		default:
			pids, err := m.Predict(ids)
			if err != nil {
				return err
			}
			numtok += 1
			if pids[0] == m.EotId || pids[0] == -1 {
				buf.FormatAdd("\n") // Format the remaining string if any
				elapsed := time.Since(start)
				fmt.Printf("\n[%d Tokens generated, %.1f tokens/s]\n", numtok, float64(numtok)/elapsed.Seconds())
				return nil
			}
			ids[0] = append(ids[0], pids[0])
			str := m.Decode(pids[0])
			if utf8.ValidString(str) {
				buf.FormatAdd(str)
				continue
			}
			// Llama3 model may output incomplete utf8 string, sigh ...
			part.WriteString(str)
			s := part.String()
			if utf8.ValidString(s) {
				buf.FormatAdd(s)
				part.Reset()
			}
		}
	}
}

func (m *Model) Setup() error {
	e := m.PredicHandler.Setup(m.Predictor)
	pbar.Tick()
	return e
}

func (m *Model) Predict(ids [][]int32) ([]int32, error) {
	return m.PredicHandler.Predict(m.Predictor, ids)
}
