package llm

import (
	"fmt"
	"strings"
	"unsafe"
)

type Operator struct {
	name string
	fn   func(arrs ...*Array) (*Array, error)
}
type Tensor struct {
	*Array
	operator *Operator
	operands []*Tensor
}

// Matmul returns a new Tensor that will perform fused matrix multiplication on the operands in the forward pass.
func (t *Tensor) Matmul(b *Tensor, bias *Tensor) *Tensor {
	ret := &Tensor{}
	ret.operands = []*Tensor{t, b, bias}
	ret.operator = &Operator{"Matmul", func(arrs ...*Array) (*Array, error) {
		a, err := arrs[0].Matmul(arrs[1], arrs[2])
		if err != nil {
			return nil, err
		}
		return a, nil
	}}
	return ret
}

// Softmax returns a new Tensor that will perform softmax on the self in the forward pass.
func (t *Tensor) Softmax() *Tensor {
	ret := &Tensor{}
	ret.operands = []*Tensor{t}
	ret.operator = &Operator{"Softmax", func(arrs ...*Array) (*Array, error) {
		a, err := arrs[0].Softmax()
		if err != nil {
			return nil, err
		}
		return a, nil
	}}
	return ret
}

// Forward performs the forward pass on the Tensor in a recursive manner.
func (t *Tensor) Forward() error {
	if t.operator == nil { // leaf node
		return nil
	}
	arrs := make([]*Array, len(t.operands))
	for i, op := range t.operands {
		if op == nil {
			continue
		}
		if err := op.Forward(); err != nil {
			return err
		}
		arrs[i] = op.Array
	}
	a, err := t.operator.fn(arrs...)
	if err != nil {
		return err
	}
	t.Array = a
	return nil
}

// Data returns the data of the Tensor on the host.
func (t *Tensor) Data() any {
	if t.Array == nil {
		return nil
	}
	return t.Array.ToHost()
}

// Shape is a syntactic sugar to get the shape of the Tensor's data.
func (t *Tensor) Shape() Shape {
	return t.Array.Shape
}

func MakeTensorFrom(s Shape, p unsafe.Pointer, dt DType) (*Tensor, error) {
	a, err := MakeArrayFrom(s, p, dt)
	if err != nil {
		return nil, err
	}
	return &Tensor{Array: a}, nil
}

// MakeTensor creates a new Tensor from the given Shape and data. For e.g,
// MakeTensor(Shape{2, 2}, []float32{1, 2, 3, 4}) creates a 2D Tensor of float32.
func MakeTensor(s Shape, data any) (*Tensor, error) {
	a, err := MakeArray(s, data)
	if err != nil {
		return nil, err
	}
	return &Tensor{Array: a}, nil
}

func (t *Tensor) Format(f fmt.State, c rune) {
	switch c {
	case 'g':
		fmt.Fprint(f, drawTensor(t))
	default:
		fmt.Fprintf(f, "%v", t.Array)
	}
}

func (t *Tensor) Embedding(ids *Tensor) *Tensor {
	ret := &Tensor{}
	ret.operands = []*Tensor{t, ids}
	ret.operator = &Operator{"Embedding", func(arrs ...*Array) (*Array, error) {
		a, err := arrs[0].Embedding(arrs[1])
		if err != nil {
			return nil, err
		}
		return a, nil
	}}
	return ret
}

func (t *Tensor) Rmsnorm(x *Tensor, eps float32) *Tensor {
	ret := &Tensor{}
	ret.operands = []*Tensor{t, x}
	ret.operator = &Operator{"Rmsnorm", func(arrs ...*Array) (*Array, error) {
		a, err := arrs[0].Rmsnorm(arrs[1], eps)
		if err != nil {
			return nil, err
		}
		return a, nil
	}}
	return ret
}

func drawTensor(t *Tensor) string {
	var sb strings.Builder
	drawRecursive(t, &sb, "", true, true)
	return sb.String()
}

func drawRecursive(t *Tensor, sb *strings.Builder, prefix string, isLast bool, isRoot bool) {
	marker := "├── "
	if isLast {
		marker = "└── "
	}
	if !isRoot {
		sb.WriteString(prefix + marker)
	}

	if isRoot || t.operator != nil {
		// For root or intermediate nodes, show operator name
		opName := "None"
		if t.operator != nil {
			opName = t.operator.name
		}
		sb.WriteString(fmt.Sprintf("Tensor (%s)\n", opName))
	} else {
		// For leaf nodes, show Shape
		sb.WriteString(fmt.Sprintf("Tensor %v\n", t.Shape()))
	}

	childPrefix := prefix
	if isLast {
		childPrefix += "    "
	} else {
		childPrefix += "│   "
	}

	for i, operand := range t.operands {
		isLastChild := (i == len(t.operands)-1)
		if operand != nil {
			drawRecursive(operand, sb, childPrefix, isLastChild, false)
		}
	}
}
