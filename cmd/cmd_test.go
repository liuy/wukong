package cmd

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/liuy/wukong/assert"
)

func TestFlagSet(t *testing.T) {
	fs := NewFlagSet()
	fs.SetBool("-v", "--verbose", false, "enable verbose mode")

	err := fs.parse([]string{"--verbose"})
	assert.NoErr(t, err)

	assert.True(t, fs.GetBool("--verbose"))

	err = fs.parse([]string{"--unknown"})
	assert.Error(t, err)

	// Test integer flags
	fs = NewFlagSet()
	fs.SetInt("-n", "--number", 0, "a number flag")

	// Test valid integer input
	err = fs.parse([]string{"--number", "42"})
	assert.NoErr(t, err)
	assert.Equal(t, 42, fs.GetInt("--number"))
	assert.Equal(t, 42, fs.GetInt("-n"))

	// Test missing value for integer flag
	err = fs.parse([]string{"--number"})
	assert.Error(t, err)

	// Test invalid integer format
	err = fs.parse([]string{"--number", "not-a-number"})
	assert.Error(t, err)
}

func TestCommand(t *testing.T) {
	var called bool
	cmd := NewCommand("test", "a test command", func(ctx context.Context, c *Command) error {
		called = true
		if c.Arg(0) != "" {
			return fmt.Errorf("unexpected arg: %s", c.Arg(0))
		}
		return nil
	})

	err := cmd.Run(context.Background(), []string{"test"})
	assert.NoErr(t, err)
	assert.True(t, called)

	called = false
	subCmd := NewCommand("sub", "a sub command", func(ctx context.Context, c *Command) error {
		called = true
		return nil
	})
	cmd.SubCommand(subCmd)

	err = cmd.Run(context.Background(), []string{"test", "sub"})
	assert.NoErr(t, err)
	assert.True(t, called)

	err = cmd.Run(context.Background(), []string{"test", "unknown"})
	assert.Error(t, err)

	err = cmd.Run(context.Background(), []string{"test", "--unknown"})
	assert.Error(t, err)

	cmd.SetBool("-t", "--test", false, "a test flag")
	err = cmd.Run(context.Background(), []string{"test", "--test"})
	assert.NoErr(t, err)

	var buf bytes.Buffer
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	err = cmd.Run(context.Background(), []string{"test", "--help"})
	assert.NoErr(t, err)

	w.Close()
	os.Stdout = oldStdout
	buf.ReadFrom(r)
	str := buf.String()
	assert.True(t, strings.Contains(str, "test flag"))
}
