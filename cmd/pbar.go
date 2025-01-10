package cmd

import (
	"fmt"
)

type ProgressBar struct {
	Progress       int
	BarWidth       float64
	Max            float64
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

func (pb *ProgressBar) SetProgress(value float64) {
	pb.Progress = int(value)
	pb.PrintProgress()
}

func (pb *ProgressBar) Tick() {
	pb.Progress += 1
	pb.PrintProgress()
}

func (pb *ProgressBar) PrintProgress() {
	fmt.Print(pb.PrefixText)
	fmt.Print(pb.Start)

	pos := float64(pb.Progress) * pb.BarWidth / pb.Max
	for i := 0; i < int(pb.BarWidth); i++ {
		if i <= int(pos) {
			fmt.Print(pb.Fill)
		} else {
			fmt.Print(pb.Remainder)
		}
	}

	fmt.Print(pb.End)
	if pb.ShowPercentage {
		fmt.Printf(" %d%%", int(float64(pb.Progress)/pb.Max*100.0))
	} else {
		fmt.Printf(" %d/%d", pb.Progress, int(pb.Max))
	}
	fmt.Print(" ", pb.PostfixText, "\r")

	if pb.Progress >= int(pb.Max) {
		pb.Progress = 0
		fmt.Print("\033[2K\r") // Erase line
	}
}

func HideCursor() {
	fmt.Print("\033[?25l")
}

func ShowCursor() {
	fmt.Print("\033[?25h")
}
