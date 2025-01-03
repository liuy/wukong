package llm

type Config struct {
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
}

type Model struct {
	*Config
	*Tokenizer
}

func NewModel(path string) (*Model, error) {
	gguf, err := GGUFParser(path)
	if err != nil {
		return nil, err
	}
	return &Model{
		Config:    gguf.GetConfig(),
		Tokenizer: gguf.GetTokenizer(),
	}, nil
}
