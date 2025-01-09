package main

import (
	"context"
	"fmt"
	"os"

	"github.com/liuy/wukong/cmd"
)

var version = "0.1"

func main() {
	m := cmd.NewCommand("wk", "A simple cmdline for wukong",
		func(ctx context.Context, c *cmd.Command) error {
			if c.GetBool("--version") {
				fmt.Println("Version:", version)
				return nil
			}
			c.ShowHelp()
			return nil
		})

	run := cmd.NewCommand("run", "Run the model",
		func(ctx context.Context, c *cmd.Command) error {
			if c.Arg(0) == "" {
				return fmt.Errorf("missing model name")
			}
			fmt.Println("Run model:", c.Arg(0))
			return nil
		})

	m.SubCommand(run)
	m.SetBool("-v", "--version", false, "show version")

	if err := m.Run(context.Background(), os.Args); err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}
}
