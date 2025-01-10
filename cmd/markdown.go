package cmd

import (
	"bytes"
	"strings"
)

const (
	BoldStart     = "\x1b[1m"
	BoldEnd       = "\x1b[22m"
	ItalicStart   = "\x1b[3m"
	ItalicEnd     = "\x1b[23m"
	CodeStart     = "\x1b[7m"        // inverse colors
	CodeEnd       = "\x1b[27m"       // reset inverse colors
	Header1Start  = "\x1b[1;32m"     // green
	Header2Start  = "\x1b[1;36m"     // cyan
	Header3Start  = "\x1b[1;34m"     // blue
	HeaderEnd     = "\x1b[0m"        // reset all
	BlockQuote    = "\x1b[38;5;240m" // gray
	BlockQuoteEnd = "\x1b[0m"        // reset all
	LinkStart     = "\x1b[4;34m"     // underline blue
	LinkEnd       = "\x1b[24;39m"    // reset underline and color
)

type MarkdownFormatter struct {
	buf         *bytes.Buffer
	inCode      bool
	inCodeBlock bool
	inBold      bool
	inItalic    bool
}

func NewMarkdownFormatter() *MarkdownFormatter {
	return &MarkdownFormatter{
		buf: bytes.NewBuffer(nil),
	}
}

func (m *MarkdownFormatter) Format(input string) string {
	m.buf.Reset()
	lines := strings.Split(input, "\n") // Note For e.g, "text\n\n" -> ["text", "", ""]

	// remove the extra "" line added by the split if input ends with a newline
	if len(lines) > 1 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}

	for _, line := range lines {
		m.formatLine(line)
		m.buf.WriteByte('\n')
	}

	return m.buf.String()
}

func (m *MarkdownFormatter) formatLine(line string) {
	// Handle code blocks
	if strings.HasPrefix(line, "```") { // Just remove "```" from the line
		if m.inCodeBlock {
			m.inCodeBlock = false
		} else {
			m.inCodeBlock = true
		}
		return
	}

	// If we're in a code block, preserve the line as-is
	if m.inCodeBlock {
		m.buf.WriteString(line)
		return
	}

	// Handle headers
	if strings.HasPrefix(line, "# ") {
		m.buf.WriteString(Header1Start)
		m.formatInline(strings.TrimPrefix(line, "# "))
		m.buf.WriteString(HeaderEnd)
		return
	}
	if strings.HasPrefix(line, "## ") {
		m.buf.WriteString(Header2Start)
		m.formatInline(strings.TrimPrefix(line, "## "))
		m.buf.WriteString(HeaderEnd)
		return
	}
	if strings.HasPrefix(line, "### ") {
		m.buf.WriteString(Header3Start)
		m.formatInline(strings.TrimPrefix(line, "### "))
		m.buf.WriteString(HeaderEnd)
		return
	}

	// Handle blockquotes
	if strings.HasPrefix(line, ">") {
		m.buf.WriteString(BlockQuote)
		m.buf.WriteString(line)
		m.buf.WriteString(BlockQuoteEnd)
		return
	}

	// Handle inline formatting
	m.formatInline(line)
}

func (m *MarkdownFormatter) formatInline(text string) {
	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		switch {
		case runes[i] == '*':
			switch {
			case i+1 < len(runes) && runes[i+1] == ' ': // * bullet list
				m.buf.WriteString("  • ")
				i++ // skip the space
			case i+1 < len(runes) && runes[i+1] == '*': // **bold**
				if m.inBold {
					m.buf.WriteString(BoldEnd)
					m.inBold = false
				} else {
					m.buf.WriteString(BoldStart)
					m.inBold = true
				}
				i++ // skip the next '*'
			case i+1 < len(runes) && runes[i+1] == '_': // *_italic_*
				if m.inItalic {
					m.buf.WriteString(ItalicEnd)
					m.inItalic = false
				} else {
					m.buf.WriteString(ItalicStart)
					m.inItalic = true
				}
				i++ // skip the next '_'
			default: // default to italic
				if m.inItalic {
					m.buf.WriteString(ItalicEnd)
					m.inItalic = false
				} else {
					m.buf.WriteString(ItalicStart)
					m.inItalic = true
				}
			}
		case runes[i] == '_': // _italic_
			switch {
			case i+1 < len(runes) && runes[i+1] == '*': // *_italic_*
				if m.inItalic {
					m.buf.WriteString(ItalicEnd)
					m.inItalic = false
				} else {
					m.buf.WriteString(ItalicStart)
					m.inItalic = true
				}
				i++ // skip the next '*'
			case i+1 < len(runes) && runes[i+1] == '_': // __bold__
				if m.inBold {
					m.buf.WriteString(BoldEnd)
					m.inBold = false
				} else {
					m.buf.WriteString(BoldStart)
					m.inBold = true
				}
				i++ // skip the next '_'
			default: // default to italic
				if m.inItalic {
					m.buf.WriteString(ItalicEnd)
					m.inItalic = false
				} else {
					m.buf.WriteString(ItalicStart)
					m.inItalic = true
				}
			}
		case runes[i] == '`':
			if m.inCode {
				m.buf.WriteString(CodeEnd)
				m.inCode = false
			} else {
				m.buf.WriteString(CodeStart)
				m.inCode = true
			}
		case runes[i] == '[' && runesContains(runes[i:], "]("):
			// Extract link text and URL
			start := i + 1
			textEnd := start
			for textEnd < len(runes) && runes[textEnd] != ']' {
				textEnd++
			}
			linkText := string(runes[start:textEnd])

			urlStart := textEnd + 2 // skip ](
			urlEnd := urlStart
			for urlEnd < len(runes) && runes[urlEnd] != ')' {
				urlEnd++
			}
			linkURL := string(runes[urlStart:urlEnd])

			// Write terminal hyperlink
			m.buf.WriteString(LinkStart)
			m.buf.WriteString("\x1b]8;;")
			m.buf.WriteString(linkURL)
			m.buf.WriteString("\x1b\\")
			m.buf.WriteString(linkText)
			m.buf.WriteString("\x1b]8;;\x1b\\")
			m.buf.WriteString(LinkEnd)

			// Advance iterator past the entire link
			i = urlEnd
		default:
			m.buf.WriteRune(runes[i])
		}
	}

	// Reset any open formatting
	if m.inBold {
		m.buf.WriteString(BoldEnd)
		m.inBold = false
	}
	if m.inItalic {
		m.buf.WriteString(ItalicEnd)
		m.inItalic = false
	}
	if m.inCode {
		m.buf.WriteString(CodeEnd)
		m.inCode = false
	}
	if m.inCodeBlock {
		m.buf.WriteString(CodeEnd)
		m.inCodeBlock = false
	}
}

func runesContains(runes []rune, substr string) bool {
	s := string(runes)
	return strings.Contains(s, substr)
}
