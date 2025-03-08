package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/liuy/wukong/cmd"
	"github.com/liuy/wukong/llm"
)

type MultilineState int

const (
	MultilineNone MultilineState = iota
	MultilinePrompt
	MultilineSystem
)

func runHandler(ctx context.Context, c *cmd.Command) error {
	if c.Arg(0) == "" {
		return fmt.Errorf("missing model name")
	}
	idx := c.GetInt("--gpu")
	if idx < 0 {
		return fmt.Errorf("invalid GPU index")
	}
	llm.CudaSetup(idx)
	defer llm.CudaTeardown()

	model, err := llm.NewModel(ctx, c.Arg(0))
	if err != nil {
		return err
	}
	err = model.Setup()
	if err != nil {
		return err
	}

	scanner, err := cmd.NewTerm(cmd.Prompt{
		Prompt:         ">>> ",
		AltPrompt:      "... ",
		Placeholder:    "Hi there! What can I do for you? (/? for help)",
		AltPlaceholder: `Use """ to end multi-line input`,
	})
	if err != nil {
		return err
	}
	fmt.Print(cmd.StartBracketedPaste)
	defer fmt.Printf(cmd.EndBracketedPaste)

	var sb strings.Builder
	var multiline MultilineState

	for {
		line, err := scanner.Readline()
		switch {
		case errors.Is(err, io.EOF):
			fmt.Println()
			return nil
		case errors.Is(err, cmd.ErrInterrupt):
			if line == "" {
				fmt.Println("\nUse Ctrl + d or /bye to exit.")
			}

			scanner.Prompt.UseAlt = false
			sb.Reset()

			continue
		case err != nil:
			return err
		}

		switch {
		case multiline != MultilineNone:
			// check if there's a multiline terminating string
			before, ok := strings.CutSuffix(line, `"""`)
			sb.WriteString(before)
			if !ok {
				fmt.Fprintln(&sb)
				continue
			}

			multiline = MultilineNone
			scanner.Prompt.UseAlt = false
		case strings.HasPrefix(line, `"""`):
			line := strings.TrimPrefix(line, `"""`)
			line, ok := strings.CutSuffix(line, `"""`)
			sb.WriteString(line)
			if !ok {
				// no multiline terminating string; need more input
				fmt.Fprintln(&sb)
				multiline = MultilinePrompt
				scanner.Prompt.UseAlt = true
			}
		case scanner.Pasting:
			fmt.Fprintln(&sb, line)
			continue
		case strings.HasPrefix(line, "/help"), strings.HasPrefix(line, "/?"):
			args := strings.Fields(line)
			if len(args) > 1 {
				switch args[1] {
				case "shorts", "shortcut", "shortcuts":
					fmt.Fprintln(os.Stdout, "Available keyboard shortcuts:")
					fmt.Fprintln(os.Stdout, "  Ctrl + a            Move to the beginning of the line (Home)")
					fmt.Fprintln(os.Stdout, "  Ctrl + e            Move to the end of the line (End)")
					fmt.Fprintln(os.Stdout, "   Alt + b            Move back (left) one word")
					fmt.Fprintln(os.Stdout, "   Alt + f            Move forward (right) one word")
					fmt.Fprintln(os.Stdout, "  Ctrl + k            Delete the sentence after the cursor")
					fmt.Fprintln(os.Stdout, "  Ctrl + u            Delete the sentence before the cursor")
					fmt.Fprintln(os.Stdout, "  Ctrl + w            Delete the word before the cursor")
					fmt.Fprintln(os.Stdout, "")
					fmt.Fprintln(os.Stdout, "  Ctrl + l            Clear the screen")
					fmt.Fprintln(os.Stdout, "  Ctrl + c            Stop the model from responding")
					fmt.Fprintln(os.Stdout, "  Ctrl + d            Exit wukong (/bye)")
					fmt.Fprintln(os.Stdout, "")
				}
			} else {
				fmt.Fprintln(os.Stdout, "Available Commands:")
				fmt.Fprintln(os.Stdout, "  /?, /help       Help for a command")
				fmt.Fprintln(os.Stdout, "  /bye, /exit     Exit")
				fmt.Fprintln(os.Stdout, "  /? shorts       Help for keyboard shortcuts")
				fmt.Fprintln(os.Stdout, "")
				fmt.Fprintln(os.Stdout, "Use \"\"\" to begin a multi-line message.")
				fmt.Fprintln(os.Stdout, "")
			}
		case strings.HasPrefix(line, "/exit"), strings.HasPrefix(line, "/bye"):
			return nil
		default:
			sb.WriteString(line)
		}

		if sb.Len() > 0 && multiline == MultilineNone {
			msg := map[string]string{
				"system": "You are a helpful assistant.",
				"user":   sb.String(),
			}
			err := model.Generate(msg)
			if err != nil {
				return err
			}
			sb.Reset()
		}
	}
}

func main() {
	m := cmd.NewCommand("wk", "A simple cmdline for wukong",
		func(ctx context.Context, c *cmd.Command) error {
			if c.GetBool("--version") {
				fmt.Println("Version:", llm.Version)
				return nil
			}
			c.ShowHelp()
			return nil
		})

	run := cmd.NewCommand("run", "Run the model", runHandler)
	run.SetInt("-g", "--gpu", 0, "GPU device to use")

	pull := cmd.NewCommand("pull", "Pull the model", func(ctx context.Context, c *cmd.Command) error {
		if c.Arg(0) == "" {
			return fmt.Errorf("missing model name")
		}
		_, err := llm.PullModel(ctx, c.Arg(0))
		return err
	})

	m.SubCommand(
		run,
		pull,
	)
	m.SetBool("-v", "--version", false, "show version")

	if err := m.Run(context.Background(), os.Args); err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}
}
