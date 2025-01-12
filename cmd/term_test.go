package cmd

import (
	"testing"

	"github.com/liuy/wukong/assert"
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

func TestMarkdownFormatter(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "headers",
			input:    "# Header1\n## Header2\n### Header3",
			expected: Header1Start + "Header1" + HeaderEnd + "\n" + Header2Start + "Header2" + HeaderEnd + "\n" + Header3Start + "Header3" + HeaderEnd + "\n",
		},
		{
			name:     "bold text",
			input:    "**bold** text",
			expected: BoldStart + "bold" + BoldEnd + " text\n",
		},
		{
			name:     "italic text",
			input:    "*italic* text",
			expected: ItalicStart + "italic" + ItalicEnd + " text\n",
		},
		{
			name:     "bullet list",
			input:    "* bullet 1\n* bullet 2",
			expected: "  • bullet 1\n  • bullet 2\n",
		},
		{
			name:     "code",
			input:    "`code`",
			expected: CodeStart + "code" + CodeEnd + "\n",
		},
		{
			name:     "blockquote",
			input:    "> quoted text\n>\n",
			expected: BlockQuote + "> quoted text" + BlockQuoteEnd + "\n" + BlockQuote + ">" + BlockQuoteEnd + "\n",
		},
		{
			name:     "link",
			input:    "[text](https://example.com)",
			expected: LinkStart + "\x1b]8;;https://example.com\x1b\\text\x1b]8;;\x1b\\" + LinkEnd + "\n",
		},
		{
			name:     "code block",
			input:    "```\ncode block\n```",
			expected: "\ncode block\n\n",
		},
		{
			name:     "mixed formatting",
			input:    "**bold** and *italic* and `code`",
			expected: BoldStart + "bold" + BoldEnd + " and " + ItalicStart + "italic" + ItalicEnd + " and " + CodeStart + "code" + CodeEnd + "\n",
		},
	}

	formatter := NewMarkdownFormatter()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatter.Format(tt.input)
			assert.Equal(t, tt.expected, got)
		})
	}
}
