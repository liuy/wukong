package llm

import (
	"os"
	"path/filepath"
	"strings"
)

func Var(key string) string {
	return strings.Trim(strings.TrimSpace(os.Getenv(key)), "\"'")
}

func ModelsPath() string {
	if s := Var("WUKONG_MODELS"); s != "" {
		return s
	}

	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return filepath.Join(home, ".wukong", "models")
}
