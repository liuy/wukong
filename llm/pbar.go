package llm

import (
	"fmt"
)

const (
	Byte = 1

	KiloByte = Byte * 1024
	MegaByte = KiloByte * 1024
	GigaByte = MegaByte * 1024
	TeraByte = GigaByte * 1024
)

func HumanBytes(b int64) string {
	switch {
	case b >= TeraByte:
		return fmt.Sprintf("%.1f TB", float64(b)/TeraByte)
	case b >= GigaByte:
		return fmt.Sprintf("%.1f GB", float64(b)/GigaByte)
	case b >= MegaByte:
		return fmt.Sprintf("%.1f MB", float64(b)/MegaByte)
	case b >= KiloByte:
		return fmt.Sprintf("%.1f KB", float64(b)/KiloByte)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

type ProgressBar struct {
	Progress       int64
	BarWidth       float64
	Max            int64
	ShowPercentage bool
	PrefixText     string
	Start          string
	Fill           string
	Remainder      string
	End            string
	PostfixText    string
}

func NewProgressBar() *ProgressBar {
	return &ProgressBar{
		Progress:       0,
		BarWidth:       50,
		Max:            100.0,
		ShowPercentage: false,
		PrefixText:     "",
		Start:          "[",
		Fill:           "■",
		Remainder:      " ",
		End:            "]",
		PostfixText:    "",
	}
}

func (pb *ProgressBar) Reset() {
	pb.ShowPercentage = false
	pb.Progress = 0
	pb.PrefixText = ""
	pb.PostfixText = ""
}

func (pb *ProgressBar) SetProgress(value int64) {
	pb.Progress = value
	pb.PrintProgress()
}

func (pb *ProgressBar) Tick() {
	pb.Progress += 1
	pb.PrintProgress()
}

func (pb *ProgressBar) PrintProgress() {
	HideCursor()
	fmt.Print(pb.PrefixText)
	fmt.Print(pb.Start)

	pos := float64(pb.Progress) * pb.BarWidth / float64(pb.Max)
	for i := 0; i < int(pb.BarWidth); i++ {
		if i <= int(pos) {
			fmt.Print(pb.Fill)
		} else {
			fmt.Print(pb.Remainder)
		}
	}

	fmt.Print(pb.End)
	if pb.ShowPercentage {
		fmt.Printf(" %d%%", int(float64(pb.Progress)/float64(pb.Max)*100.0))
	} else {
		fmt.Printf("%10s/%s", HumanBytes(pb.Progress), HumanBytes(pb.Max))
	}
	fmt.Print(" ", pb.PostfixText, "\r")

	if pb.Progress >= int64(pb.Max) {
		pb.Progress = 0
		fmt.Print("\033[2K\r") // Erase line
	}
	ShowCursor()
}

func HideCursor() {
	fmt.Print("\033[?25l")
}

func ShowCursor() {
	fmt.Print("\033[?25h")
}
