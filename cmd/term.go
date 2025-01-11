package cmd

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"unicode"
	"unicode/utf8"
)

type Termios syscall.Termios

const (
	CharNull      = 0
	CharLineStart = 1
	CharBackward  = 2
	CharInterrupt = 3
	CharDelete    = 4
	CharLineEnd   = 5
	CharForward   = 6
	CharBell      = 7
	CharCtrlH     = 8
	CharTab       = 9
	CharCtrlJ     = 10
	CharKill      = 11
	CharCtrlL     = 12
	CharEnter     = 13
	CharNext      = 14
	CharPrev      = 16
	CharBckSearch = 18
	CharFwdSearch = 19
	CharTranspose = 20
	CharCtrlU     = 21
	CharCtrlW     = 23
	CharCtrlY     = 25
	CharCtrlZ     = 26
	CharEsc       = 27
	CharSpace     = 32
	CharEscapeEx  = 91
	CharBackspace = 127
)

const (
	KeyDel    = 51
	KeyUp     = 65
	KeyDown   = 66
	KeyRight  = 67
	KeyLeft   = 68
	MetaEnd   = 70
	MetaStart = 72
)

const (
	Esc = "\x1b"

	CursorSave    = Esc + "[s"
	CursorRestore = Esc + "[u"

	CursorEOL  = Esc + "[E"
	CursorBOL  = Esc + "[1G"
	CursorHide = Esc + "[?25l"
	CursorShow = Esc + "[?25h"

	ClearToEOL  = Esc + "[K"
	ClearLine   = Esc + "[2K"
	ClearScreen = Esc + "[2J"
	CursorReset = Esc + "[0;0f"

	ColorGrey    = Esc + "[38;5;245m"
	ColorDefault = Esc + "[0m"

	StartBracketedPaste = Esc + "[?2004h"
	EndBracketedPaste   = Esc + "[?2004l"
)

func CursorUpN(n int) string {
	return Esc + "[" + strconv.Itoa(n) + "A"
}

func CursorDownN(n int) string {
	return Esc + "[" + strconv.Itoa(n) + "B"
}

func CursorRightN(n int) string {
	return Esc + "[" + strconv.Itoa(n) + "C"
}

func CursorLeftN(n int) string {
	return Esc + "[" + strconv.Itoa(n) + "D"
}

var ErrInterrupt = errors.New("Interrupt")

type InterruptError struct {
	Line []rune
}

func (*InterruptError) Error() string {
	return "Interrupted"
}

var (
	CursorUp    = CursorUpN(1)
	CursorDown  = CursorDownN(1)
	CursorRight = CursorRightN(1)
	CursorLeft  = CursorLeftN(1)
)

const (
	CharBracketedPaste      = 50
	CharBracketedPasteStart = "00~"
	CharBracketedPasteEnd   = "01~"
)

func SetRawMode(fd uintptr) (*Termios, error) {
	termios, err := getTermios(fd)
	if err != nil {
		return nil, err
	}

	newTermios := *termios
	newTermios.Iflag &^= syscall.IGNBRK | syscall.BRKINT | syscall.PARMRK | syscall.ISTRIP | syscall.INLCR | syscall.IGNCR | syscall.ICRNL | syscall.IXON
	newTermios.Lflag &^= syscall.ECHO | syscall.ECHONL | syscall.ICANON | syscall.ISIG | syscall.IEXTEN
	newTermios.Cflag &^= syscall.CSIZE | syscall.PARENB
	newTermios.Cflag |= syscall.CS8
	newTermios.Cc[syscall.VMIN] = 1
	newTermios.Cc[syscall.VTIME] = 0

	return termios, setTermios(fd, &newTermios)
}

func UnsetRawMode(fd uintptr, termios any) error {
	t := termios.(*Termios)
	t.Lflag &^= syscall.ECHOCTL // disable echo of control characters
	return setTermios(fd, t)
}

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal(fd uintptr) bool {
	_, err := getTermios(fd)
	return err == nil
}

type Prompt struct {
	Prompt         string
	AltPrompt      string
	Placeholder    string
	AltPlaceholder string
	UseAlt         bool
}

func (p *Prompt) prompt() string {
	if p.UseAlt {
		return p.AltPrompt
	}
	return p.Prompt
}

func (p *Prompt) placeholder() string {
	if p.UseAlt {
		return p.AltPlaceholder
	}
	return p.Placeholder
}

type Terminal struct {
	outchan chan rune
	rawmode bool
	termios any
}

type Instance struct {
	Prompt   *Prompt
	Terminal *Terminal
	History  *History
	Pasting  bool
}

func NewTerm(prompt Prompt) (*Instance, error) {
	term, err := NewTerminal()
	if err != nil {
		return nil, err
	}

	history, err := NewHistory()
	if err != nil {
		return nil, err
	}

	return &Instance{
		Prompt:   &prompt,
		Terminal: term,
		History:  history,
	}, nil
}

func (i *Instance) Readline() (string, error) {
	if !i.Terminal.rawmode {
		fd := os.Stdin.Fd()
		termios, err := SetRawMode(fd)
		if err != nil {
			return "", err
		}
		i.Terminal.rawmode = true
		i.Terminal.termios = termios
	}

	prompt := i.Prompt.prompt()
	if i.Pasting {
		// force alt prompt when pasting
		prompt = i.Prompt.AltPrompt
	}
	fmt.Print(prompt)

	defer func() {
		fd := os.Stdin.Fd()
		//nolint:errcheck
		UnsetRawMode(fd, i.Terminal.termios)
		i.Terminal.rawmode = false
	}()

	buf, _ := NewBuffer(i.Prompt)

	var esc bool
	var escex bool
	var metaDel bool

	var currentLineBuf []rune

	for {
		// don't show placeholder when pasting unless we're in multiline mode
		showPlaceholder := !i.Pasting || i.Prompt.UseAlt
		if buf.IsEmpty() && showPlaceholder {
			ph := i.Prompt.placeholder()
			fmt.Print(ColorGrey + ph + CursorLeftN(len(ph)) + ColorDefault)
		}

		r, err := i.Terminal.Read()

		if buf.IsEmpty() {
			fmt.Print(ClearToEOL)
		}

		if err != nil {
			return "", io.EOF
		}

		if escex {
			escex = false

			switch r {
			case KeyUp:
				if i.History.Pos > 0 {
					if i.History.Pos == i.History.Size() {
						currentLineBuf = []rune(buf.String())
					}
					buf.Replace(i.History.Prev())
				}
			case KeyDown:
				if i.History.Pos < i.History.Size() {
					buf.Replace(i.History.Next())
					if i.History.Pos == i.History.Size() {
						buf.Replace(currentLineBuf)
					}
				}
			case KeyLeft:
				buf.MoveLeft()
			case KeyRight:
				buf.MoveRight()
			case CharBracketedPaste:
				var code string
				for range 3 {
					r, err = i.Terminal.Read()
					if err != nil {
						return "", io.EOF
					}

					code += string(r)
				}
				if code == CharBracketedPasteStart {
					i.Pasting = true
				} else if code == CharBracketedPasteEnd {
					i.Pasting = false
				}
			case KeyDel:
				if buf.DisplaySize() > 0 {
					buf.Delete()
				}
				metaDel = true
			case MetaStart:
				buf.MoveToStart()
			case MetaEnd:
				buf.MoveToEnd()
			default:
				// skip any keys we don't know about
				continue
			}
			continue
		} else if esc {
			esc = false

			switch r {
			case 'b':
				buf.MoveLeftWord()
			case 'f':
				buf.MoveRightWord()
			case CharBackspace:
				buf.DeleteWord()
			case CharEscapeEx:
				escex = true
			}
			continue
		}

		switch r {
		case CharNull:
			continue
		case CharEsc:
			esc = true
		case CharInterrupt:
			return "", ErrInterrupt
		case CharLineStart:
			buf.MoveToStart()
		case CharLineEnd:
			buf.MoveToEnd()
		case CharBackward:
			buf.MoveLeft()
		case CharForward:
			buf.MoveRight()
		case CharBackspace, CharCtrlH:
			buf.Remove()
		case CharTab:
			// todo: convert back to real tabs
			for range 8 {
				buf.Add(' ')
			}
		case CharDelete:
			if buf.DisplaySize() > 0 {
				buf.Delete()
			} else {
				return "", io.EOF
			}
		case CharKill:
			buf.DeleteRemaining()
		case CharCtrlU:
			buf.DeleteBefore()
		case CharCtrlL:
			buf.ClearScreen()
		case CharCtrlW:
			buf.DeleteWord()
		case CharCtrlZ:
			fd := os.Stdin.Fd()
			return handleCharCtrlZ(fd, i.Terminal.termios)
		case CharEnter, CharCtrlJ:
			output := buf.String()
			if output != "" {
				i.History.Add([]rune(output))
			}
			buf.MoveToEnd()
			fmt.Println()

			return output, nil
		default:
			if metaDel {
				metaDel = false
				continue
			}
			if r >= CharSpace || r == CharEnter || r == CharCtrlJ {
				buf.Add(r)
			}
		}
	}
}

func (i *Instance) HistoryEnable() {
	i.History.Enabled = true
}

func (i *Instance) HistoryDisable() {
	i.History.Enabled = false
}

func NewTerminal() (*Terminal, error) {
	fd := os.Stdin.Fd()
	termios, err := SetRawMode(fd)
	if err != nil {
		return nil, err
	}

	t := &Terminal{
		outchan: make(chan rune),
		rawmode: true,
		termios: termios,
	}

	go t.ioloop()

	return t, nil
}

func (t *Terminal) ioloop() {
	buf := bufio.NewReader(os.Stdin)

	for {
		r, _, err := buf.ReadRune()
		if err != nil {
			close(t.outchan)
			break
		}
		t.outchan <- r
	}
}

func (t *Terminal) Read() (rune, error) {
	r, ok := <-t.outchan
	if !ok {
		return 0, io.EOF
	}

	return r, nil
}

type History struct {
	Buf      [][]rune
	Autosave bool
	Pos      int
	Limit    int
	Filename string
	Enabled  bool
}

func NewHistory() (*History, error) {
	h := &History{
		Buf:      make([][]rune, 0),
		Limit:    100,
		Autosave: true,
		Enabled:  true,
	}

	err := h.Init()
	if err != nil {
		return nil, err
	}

	return h, nil
}

func (h *History) Init() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	path := filepath.Join(home, ".wukong", "history")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	h.Filename = path

	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDONLY, 0o600)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}

		line = strings.TrimSpace(line)
		if len(line) == 0 {
			continue
		}

		h.Add([]rune(line))
	}

	return nil
}

func (h *History) Add(l []rune) {
	h.Buf = append(h.Buf, l)
	h.Compact()
	h.Pos = h.Size()
	if h.Autosave {
		_ = h.Save()
	}
}

func (h *History) Compact() {
	s := len(h.Buf)
	if s > h.Limit {
		h.Buf = h.Buf[s-h.Limit:]
	}
}

func (h *History) Clear() {
	h.Buf = make([][]rune, 0)
}

func (h *History) Prev() []rune {
	if h.Pos > 0 {
		h.Pos -= 1
	}
	if h.Pos < len(h.Buf) {
		return h.Buf[h.Pos]
	}
	return nil
}

func (h *History) Next() []rune {
	if h.Pos < len(h.Buf)-1 {
		h.Pos += 1
		return h.Buf[h.Pos]
	}
	return nil
}

func (h *History) Size() int {
	return len(h.Buf)
}

func (h *History) Save() error {
	if !h.Enabled {
		return nil
	}

	tmpFile := h.Filename + ".tmp"

	f, err := os.OpenFile(tmpFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()

	buf := bufio.NewWriter(f)
	for _, line := range h.Buf {
		if _, err := buf.WriteString(string(line) + "\n"); err != nil {
			return err
		}
	}
	buf.Flush()
	f.Close()

	if err = os.Rename(tmpFile, h.Filename); err != nil {
		return err
	}

	return nil
}

type Buffer struct {
	DisplayPos   int
	Pos          int
	Buf          []rune
	LineHasSpace []bool
	Prompt       *Prompt
	LineWidth    int
	Width        int
	Height       int
}

func NewBuffer(prompt *Prompt) (*Buffer, error) {
	fd := int(os.Stdout.Fd())
	width, height := 80, 24
	if termWidth, termHeight, err := getSize(fd); err == nil {
		width, height = termWidth, termHeight
	}

	lwidth := width - len(prompt.prompt())

	b := &Buffer{
		DisplayPos:   0,
		Pos:          0,
		Buf:          []rune{},
		LineHasSpace: make([]bool, height),
		Prompt:       prompt,
		Width:        width,
		Height:       height,
		LineWidth:    lwidth,
	}

	return b, nil
}

func (b *Buffer) GetLineSpacing(line int) bool {
	if line < 0 || line >= len(b.LineHasSpace) {
		return false
	}
	return b.LineHasSpace[line]
}

func (b *Buffer) MoveLeft() {
	if b.Pos > 0 {
		r := b.Buf[b.Pos-1]
		rLength := RuneWidth(r)

		if b.DisplayPos%b.LineWidth == 0 {
			fmt.Print(CursorUp + CursorBOL + CursorRightN(b.Width))
			if rLength == 2 {
				fmt.Print(CursorLeft)
			}

			line := b.DisplayPos/b.LineWidth - 1
			hasSpace := b.GetLineSpacing(line)
			if hasSpace {
				b.DisplayPos -= 1
				fmt.Print(CursorLeft)
			}
		} else {
			fmt.Print(CursorLeftN(rLength))
		}

		b.Pos -= 1
		b.DisplayPos -= rLength
	}
}

func (b *Buffer) MoveLeftWord() {
	if b.Pos > 0 {
		var foundNonspace bool
		for {
			if b.Buf[b.Pos-1] == ' ' {
				if foundNonspace {
					break
				}
			} else {
				foundNonspace = true
			}
			b.MoveLeft()

			if b.Pos == 0 {
				break
			}
		}
	}
}

func (b *Buffer) MoveRight() {
	if b.Pos < len(b.Buf) {
		r := b.Buf[b.Pos]
		rLength := RuneWidth(r)
		b.Pos += 1
		hasSpace := b.GetLineSpacing(b.DisplayPos / b.LineWidth)
		b.DisplayPos += rLength

		if b.DisplayPos%b.LineWidth == 0 {
			fmt.Print(CursorDown + CursorBOL + CursorRightN(len(b.Prompt.prompt())))
		} else if (b.DisplayPos-rLength)%b.LineWidth == b.LineWidth-1 && hasSpace {
			fmt.Print(CursorDown + CursorBOL + CursorRightN(len(b.Prompt.prompt())+rLength))
			b.DisplayPos += 1
		} else if len(b.LineHasSpace) > 0 && b.DisplayPos%b.LineWidth == b.LineWidth-1 && hasSpace {
			fmt.Print(CursorDown + CursorBOL + CursorRightN(len(b.Prompt.prompt())))
			b.DisplayPos += 1
		} else {
			fmt.Print(CursorRightN(rLength))
		}
	}
}

func (b *Buffer) MoveRightWord() {
	if b.Pos < len(b.Buf) {
		for {
			b.MoveRight()
			if b.Buf[b.Pos] == ' ' {
				break
			}

			if b.Pos == len(b.Buf) {
				break
			}
		}
	}
}

func (b *Buffer) MoveToStart() {
	if b.Pos > 0 {
		currLine := b.DisplayPos / b.LineWidth
		if currLine > 0 {
			for i := 0; i < currLine; i++ {
				fmt.Print(CursorUp)
			}
		}
		fmt.Print(CursorBOL + CursorRightN(len(b.Prompt.prompt())))
		b.Pos = 0
		b.DisplayPos = 0
	}
}

func (b *Buffer) MoveToEnd() {
	if b.Pos < len(b.Buf) {
		currLine := b.DisplayPos / b.LineWidth
		totalLines := b.DisplaySize() / b.LineWidth
		if currLine < totalLines {
			for i := 0; i < totalLines-currLine; i++ {
				fmt.Print(CursorDown)
			}
			remainder := b.DisplaySize() % b.LineWidth
			fmt.Print(CursorBOL + CursorRightN(len(b.Prompt.prompt())+remainder))
		} else {
			fmt.Print(CursorRightN(b.DisplaySize() - b.DisplayPos))
		}

		b.Pos = len(b.Buf)
		b.DisplayPos = b.DisplaySize()
	}
}

func (b *Buffer) DisplaySize() int {
	sum := 0
	for _, r := range b.Buf {
		sum += RuneWidth(r)
	}
	return sum
}

func (b *Buffer) Add(r rune) {
	if b.Pos == len(b.Buf) {
		b.AddChar(r, false)
	} else {
		b.AddChar(r, true)
	}
}

func (b *Buffer) AddChar(r rune, insert bool) {
	rLength := RuneWidth(r)
	b.DisplayPos += rLength

	if b.Pos > 0 {
		if b.DisplayPos%b.LineWidth == 0 {
			fmt.Printf("%c", r)
			fmt.Printf("\n%s", b.Prompt.AltPrompt)

			if insert {
				b.LineHasSpace[b.DisplayPos/b.LineWidth-1] = false
			} else {
				b.LineHasSpace = append(b.LineHasSpace, false)
			}

		} else if b.DisplayPos%b.LineWidth < (b.DisplayPos-rLength)%b.LineWidth {
			if insert {
				fmt.Print(ClearToEOL)
			}
			fmt.Printf("\n%s", b.Prompt.AltPrompt)
			b.DisplayPos += 1
			fmt.Printf("%c", r)

			if insert {
				b.LineHasSpace[b.DisplayPos/b.LineWidth-1] = true
			} else {
				b.LineHasSpace = append(b.LineHasSpace, true)
			}
		} else {
			fmt.Printf("%c", r)
		}
	} else {
		fmt.Printf("%c", r)
	}

	if insert {
		b.Buf = append(b.Buf[:b.Pos], append([]rune{r}, b.Buf[b.Pos:]...)...)
	} else {
		b.Buf = append(b.Buf, r)
	}

	b.Pos += 1

	if insert {
		b.drawRemaining()
	}
}

func (b *Buffer) countRemainingLineWidth(place int) int {
	var sum int
	counter := -1
	var prevLen int

	for place <= b.LineWidth {
		counter += 1
		sum += prevLen
		if b.Pos+counter < len(b.Buf) {
			r := b.Buf[b.Pos+counter]
			place += RuneWidth(r)
			prevLen = len(string(r))
		} else {
			break
		}
	}

	return sum
}

func (b *Buffer) drawRemaining() {
	var place int
	remainingText := b.StringN(b.Pos)
	if b.Pos > 0 {
		place = b.DisplayPos % b.LineWidth
	}
	fmt.Print(CursorHide)

	currLineLength := b.countRemainingLineWidth(place)

	currLine := remainingText[:min(currLineLength, len(remainingText))]
	currLineSpace := StringWidth(currLine)
	remLength := StringWidth(remainingText)

	if len(currLine) > 0 {
		fmt.Print(ClearToEOL + currLine + CursorLeftN(currLineSpace))
	} else {
		fmt.Print(ClearToEOL)
	}

	lineIndex := b.DisplayPos / b.LineWidth
	if lineIndex >= 0 && lineIndex < len(b.LineHasSpace) {
		if currLineSpace != b.LineWidth-place && currLineSpace != remLength {
			b.LineHasSpace[lineIndex] = true
		} else if currLineSpace != b.LineWidth-place {
			if lineIndex+1 < len(b.LineHasSpace) {
				b.LineHasSpace = append(b.LineHasSpace[:lineIndex], b.LineHasSpace[lineIndex+1:]...)
			}
		} else {
			b.LineHasSpace[lineIndex] = false
		}
	}

	if (b.DisplayPos+currLineSpace)%b.LineWidth == 0 && currLine == remainingText {
		fmt.Print(CursorRightN(currLineSpace))
		fmt.Printf("\n%s", b.Prompt.AltPrompt)
		fmt.Print(CursorUp + CursorBOL + CursorRightN(b.Width-currLineSpace))
	}

	if remLength > currLineSpace {
		remaining := remainingText[len(currLine):]
		var totalLines int
		var displayLength int
		var lineLength int = currLineSpace

		for _, c := range remaining {
			if displayLength == 0 || (displayLength+RuneWidth(c))%b.LineWidth < displayLength%b.LineWidth {
				fmt.Printf("\n%s", b.Prompt.AltPrompt)
				totalLines += 1

				if displayLength != 0 {
					if lineLength == b.LineWidth {
						b.LineHasSpace[b.DisplayPos/b.LineWidth+totalLines-1] = false
					} else {
						b.LineHasSpace[b.DisplayPos/b.LineWidth+totalLines-1] = true
					}
				}

				lineLength = 0
			}

			displayLength += RuneWidth(c)
			lineLength += RuneWidth(c)
			fmt.Printf("%c", c)
		}
		fmt.Print(ClearToEOL + CursorUpN(totalLines) + CursorBOL + CursorRightN(b.Width-currLineSpace))

		hasSpace := b.GetLineSpacing(b.DisplayPos / b.LineWidth)

		if hasSpace && b.DisplayPos%b.LineWidth != b.LineWidth-1 {
			fmt.Print(CursorLeft)
		}
	}

	fmt.Print(CursorShow)
}

func (b *Buffer) Remove() {
	if len(b.Buf) > 0 && b.Pos > 0 {
		r := b.Buf[b.Pos-1]
		rLength := RuneWidth(r)
		hasSpace := b.GetLineSpacing(b.DisplayPos/b.LineWidth - 1)

		if b.DisplayPos%b.LineWidth == 0 {
			fmt.Print(CursorBOL + ClearToEOL + CursorUp + CursorBOL + CursorRightN(b.Width))

			if b.DisplaySize()%b.LineWidth < (b.DisplaySize()-rLength)%b.LineWidth {
				b.LineHasSpace = append(b.LineHasSpace[:b.DisplayPos/b.LineWidth-1], b.LineHasSpace[b.DisplayPos/b.LineWidth:]...)
			}

			if hasSpace {
				b.DisplayPos -= 1
				fmt.Print(CursorLeft)
			}

			if rLength == 2 {
				fmt.Print(CursorLeft + "  " + CursorLeftN(2))
			} else {
				fmt.Print(" " + CursorLeft)
			}
		} else if (b.DisplayPos-rLength)%b.LineWidth == 0 && hasSpace {
			fmt.Print(CursorBOL + ClearToEOL + CursorUp + CursorBOL + CursorRightN(b.Width))

			if b.Pos == len(b.Buf) {
				b.LineHasSpace = append(b.LineHasSpace[:b.DisplayPos/b.LineWidth-1], b.LineHasSpace[b.DisplayPos/b.LineWidth:]...)
			}
			b.DisplayPos -= 1
		} else {
			fmt.Print(CursorLeftN(rLength))
			for i := 0; i < rLength; i++ {
				fmt.Print(" ")
			}
			fmt.Print(CursorLeftN(rLength))
		}

		var eraseExtraLine bool
		if (b.DisplaySize()-1)%b.LineWidth == 0 || (rLength == 2 && ((b.DisplaySize()-2)%b.LineWidth == 0)) || b.DisplaySize()%b.LineWidth == 0 {
			eraseExtraLine = true
		}

		b.Pos -= 1
		b.DisplayPos -= rLength
		b.Buf = append(b.Buf[:b.Pos], b.Buf[b.Pos+1:]...)

		if b.Pos < len(b.Buf) {
			b.drawRemaining()
			if eraseExtraLine {
				remainingLines := (b.DisplaySize() - b.DisplayPos) / b.LineWidth
				fmt.Print(CursorDownN(remainingLines+1) + CursorBOL + ClearToEOL)
				place := b.DisplayPos % b.LineWidth
				fmt.Print(CursorUpN(remainingLines+1) + CursorRightN(place+len(b.Prompt.prompt())))
			}
		}
	}
}

func (b *Buffer) Delete() {
	if len(b.Buf) > 0 && b.Pos < len(b.Buf) {
		b.Buf = append(b.Buf[:b.Pos], b.Buf[b.Pos+1:]...)
		b.drawRemaining()
		if b.DisplaySize()%b.LineWidth == 0 {
			if b.DisplayPos != b.DisplaySize() {
				remainingLines := (b.DisplaySize() - b.DisplayPos) / b.LineWidth
				fmt.Print(CursorDownN(remainingLines) + CursorBOL + ClearToEOL)
				place := b.DisplayPos % b.LineWidth
				fmt.Print(CursorUpN(remainingLines) + CursorRightN(place+len(b.Prompt.prompt())))
			}
		}
	}
}

func (b *Buffer) DeleteBefore() {
	if b.Pos > 0 {
		for cnt := b.Pos - 1; cnt >= 0; cnt-- {
			b.Remove()
		}
	}
}

func (b *Buffer) DeleteRemaining() {
	if b.DisplaySize() > 0 && b.Pos < b.DisplaySize() {
		charsToDel := len(b.Buf) - b.Pos
		for i := 0; i < charsToDel; i++ {
			b.Delete()
		}
	}
}

func (b *Buffer) DeleteWord() {
	if len(b.Buf) > 0 && b.Pos > 0 {
		var foundNonspace bool
		for {
			if b.Buf[b.Pos-1] == ' ' {
				if !foundNonspace {
					b.Remove()
				} else {
					break
				}
			} else {
				foundNonspace = true
				b.Remove()
			}

			if b.Pos == 0 {
				break
			}
		}
	}
}

func (b *Buffer) ClearScreen() {
	fmt.Print(ClearScreen + CursorReset + b.Prompt.prompt())
	if b.IsEmpty() {
		ph := b.Prompt.placeholder()
		fmt.Print(ColorGrey + ph + CursorLeftN(len(ph)) + ColorDefault)
	} else {
		currPos := b.DisplayPos
		currIndex := b.Pos
		b.Pos = 0
		b.DisplayPos = 0
		b.drawRemaining()
		fmt.Print(CursorReset + CursorRightN(len(b.Prompt.prompt())))
		if currPos > 0 {
			targetLine := currPos / b.LineWidth
			if targetLine > 0 {
				for i := 0; i < targetLine; i++ {
					fmt.Print(CursorDown)
				}
			}
			remainder := currPos % b.LineWidth
			if remainder > 0 {
				fmt.Print(CursorRightN(remainder))
			}
			if currPos%b.LineWidth == 0 {
				fmt.Print(CursorBOL + b.Prompt.AltPrompt)
			}
		}
		b.Pos = currIndex
		b.DisplayPos = currPos
	}
}

func (b *Buffer) IsEmpty() bool {
	return len(b.Buf) == 0
}

func (b *Buffer) Replace(r []rune) {
	b.DisplayPos = 0
	b.Pos = 0
	lineNums := b.DisplaySize() / b.LineWidth

	b.Buf = []rune{}

	fmt.Print(CursorBOL + ClearToEOL)

	for i := 0; i < lineNums; i++ {
		fmt.Print(CursorUp + CursorBOL + ClearToEOL)
	}

	fmt.Print(CursorBOL + b.Prompt.prompt())

	for _, c := range r {
		b.Add(c)
	}
}

func (b *Buffer) String() string {
	return b.StringN(0)
}

func (b *Buffer) StringN(n int) string {
	return b.StringNM(n, 0)
}

func (b *Buffer) StringNM(n, m int) string {
	if m == 0 {
		m = len(b.Buf)
	}
	return string(b.Buf[n:m])
}

const (
	_leading  = 0xd800
	_trailing = 0xdc00
	_maxRune  = 0x10ffff
)

func RuneWidth(r rune) int {
	// Invalid runes return width 0
	if r == utf8.RuneError || r < 0 {
		return 0
	}

	// ASCII characters have width 1
	if r < 0x7F {
		return 1
	}

	// Surrogate pairs
	if r >= _leading && r < _trailing {
		return 0
	}
	if r > _maxRune {
		return 0
	}

	// Unicode properties for width
	switch {
	case unicode.IsControl(r):
		return 0
	case unicode.Is(unicode.Co, r): // Private use
		return 1
	case unicode.Is(unicode.Mn, r): // Non-spacing marks
		return 0
	case unicode.Is(unicode.Me, r): // Enclosing marks
		return 0
	case unicode.Is(unicode.Cf, r): // Format characters
		return 0
	}

	// East Asian Width
	switch {
	case r >= 0x1100 && r <= 0x115F: // Hangul Jamo
		return 2
	case r >= 0x2E80 && r <= 0x9FFF: // CJK
		return 2
	case r >= 0xAC00 && r <= 0xD7A3: // Hangul Syllables
		return 2
	case r >= 0xF900 && r <= 0xFAFF: // CJK Compatibility Ideographs
		return 2
	case r >= 0xFE10 && r <= 0xFE19: // Vertical forms
		return 2
	case r >= 0xFE30 && r <= 0xFE6F: // CJK Compatibility Forms
		return 2
	case r >= 0xFF00 && r <= 0xFF60: // Fullwidth Forms
		return 2
	case r >= 0xFFE0 && r <= 0xFFE6: // Fullwidth Forms
		return 2
	}

	return 1
}

func StringWidth(s string) int {
	width := 0
	for _, r := range s {
		width += RuneWidth(r)
	}
	return width
}
