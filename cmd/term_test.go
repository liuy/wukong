package cmd

import (
	"testing"
)

func BenchmarkFormat(b *testing.B) {
	formatter := NewMarkdownFormatter()
	input := `
# Header 1
## Header 2
### Header 3
*italic* **bold** ***bold italic***
> Blockquote
[link](https://example.com)
some text
more text
`
	for i := 0; i < b.N; i++ {
		formatter.Format(input)
	}
}
