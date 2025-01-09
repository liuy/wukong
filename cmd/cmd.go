package cmd

import (
	"context"
	"errors"
	"fmt"
)

type FlagSet struct {
	flags map[string]*Flag
	args  []string
}

type Flag struct {
	name  string
	short string
	desc  string
	value any
}

func NewFlagSet() *FlagSet {
	return &FlagSet{
		flags: make(map[string]*Flag),
	}
}

func (f *FlagSet) SetBool(short, name string, val bool, desc string) {
	flag := &Flag{
		name:  name,
		short: short,
		desc:  desc,
		value: val,
	}
	f.flags[short] = flag
	f.flags[name] = flag
}

func (f *FlagSet) parse(args []string) error {
	f.args = args
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg[0] != '-' {
			break
		}
		if flag, ok := f.flags[arg]; ok {
			flag.value = true
		} else {
			msg := fmt.Sprintf("unknown flag %s, available flags:\n\n", arg)
			seen := make(map[*Flag]bool)
			for _, v := range f.flags {
				if !seen[v] {
					msg += fmt.Sprintf("    %s, %s %s\n", v.short, v.name, v.desc)
					seen[v] = true
				}
			}
			return errors.New(msg)
		}
		f.args = f.args[1:] // remove flag from args
	}
	return nil
}

func (f *FlagSet) GetBool(name string) bool {
	if flag, ok := f.flags[name]; ok {
		return flag.value.(bool)
	}
	return false
}

func (f *FlagSet) NArg() int {
	count := 0
	for _, arg := range f.args {
		if arg[0] != '-' {
			count++
		}
	}
	return count
}

func (f *FlagSet) Arg(n int) string {
	count := -1
	for _, arg := range f.args {
		if arg[0] != '-' {
			count++
			if count == n {
				return arg
			}
		}
	}
	return ""
}

type Command struct {
	Name     string
	Desc     string
	Func     func(context.Context, *Command) error
	Commands []*Command
	*FlagSet
}

func (c *Command) Run(ctx context.Context, args []string) error {
	fs := c.FlagSet
	if err := fs.parse(args[1:]); err != nil {
		return err
	}
	if fs.GetBool("--help") {
		c.ShowHelp()
		return nil
	}

	if fs.NArg() > 0 {
		for _, sub := range c.Commands {
			if sub.Name == fs.Arg(0) {
				return sub.Run(ctx, fs.args)
			}
		}
	}

	return c.Func(ctx, c)
}

func (c *Command) ShowHelp() {
	fmt.Printf("Name:\n")
	fmt.Printf("    %s - %s\n\n", c.Name, c.Desc)
	fmt.Printf("Usage:\n")
	if len(c.Commands) > 0 {
		fmt.Printf("    %s [flags] command\n\n", c.Name)
		fmt.Printf("Available Commands:\n")
		for _, sub := range c.Commands {
			fmt.Printf("    %s - %s\n", sub.Name, sub.Desc)
		}
		fmt.Println()
	} else {
		fmt.Printf("    %s [flags] [args]\n\n", c.Name)
	}
	fmt.Printf("Flags:\n")
	seen := make(map[*Flag]bool)
	for _, v := range c.flags {
		if !seen[v] {
			fmt.Printf("    %s, %s %s\n", v.short, v.name, v.desc)
			seen[v] = true
		}
	}
}

func NewCommand(name, desc string, fn func(context.Context, *Command) error) *Command {
	c := &Command{
		Name: name,
		Desc: desc,
		Func: fn,
	}
	c.FlagSet = NewFlagSet()
	c.SetBool("-h", "--help", false, "show help")
	return c
}

func (c *Command) SubCommand(subs ...*Command) {
	for _, sub := range subs {
		if sub == c {
			panic("command can't add itself")
		}
		c.Commands = append(c.Commands, sub)
	}
}
