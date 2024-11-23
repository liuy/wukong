package tensor

import (
	"fmt"
	"testing"

	"github.com/liuy/wukong/assert"
)

func TestMakeTensor(t *testing.T) {
	shape := Shape{2, 2}
	data := []float32{1, 2, 3, 4}
	tensor, err := MakeTensor(shape, data)
	assert.NoErr(t, err)
	assert.NotNil(t, tensor)
	assert.Equal(t, shape, tensor.Shape())
	assert.Equal(t, data, tensor.Data())

	_, err = MakeTensor(Shape{2, 3}, []int8{1, 2, 3, 4})
	assert.Error(t, err)
}

func TestTensorForwardNoOperator(t *testing.T) {
	shape := Shape{2, 2}
	data := []float32{1, 2, 3, 4}
	tensor, err := MakeTensor(shape, data)
	assert.NoErr(t, err)
	assert.NotNil(t, tensor)

	err = tensor.Forward()
	assert.NoErr(t, err)
	assert.Equal(t, data, tensor.Data())
}

func TestTensorOperatorFail(t *testing.T) {
	fail := &Tensor{}
	fail.operator = &Operator{"Fail", func(arrs ...*Array) (*Array, error) {
		return nil, fmt.Errorf("failed")
	}}
	s := fail.Softmax()
	err := s.Forward()
	assert.Error(t, err)
}

func TestTensorMatmul(t *testing.T) {
	a, err := MakeTensor(Shape{2, 2}, []float32{1, 2, 3, 4})
	assert.NoErr(t, err)
	b, err := MakeTensor(Shape{2, 2}, []float32{5, 6, 7, 8})
	assert.NoErr(t, err)
	bias, err := MakeTensor(Shape{2}, []float32{1, 1})
	assert.NoErr(t, err)

	res := a.Matmul(b, bias)
	assert.NotNil(t, res)
	assert.Equal(t, "Matmul", res.operator.name)
	assert.Equal(t, []*Tensor{a, b, bias}, res.operands)
	err = res.Forward()
	assert.NoErr(t, err)
	assert.Equal(t, []float32{18, 24, 40, 54}, res.Data())

	r2 := a.Matmul(b, nil)
	assert.NotNil(t, r2)
	assert.Nil(t, r2.Data())
	err = r2.Forward()
	assert.NoErr(t, err)
	assert.Equal(t, []float32{17, 23, 39, 53}, r2.Data())

	c, _ := MakeTensor(Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	err = a.Matmul(c, nil).Forward()
	assert.Error(t, err)
}

func TestTensorSoftmax(t *testing.T) {
	shape := Shape{2, 2}
	data := []float32{1, 1, 1, 1}
	tensor, err := MakeTensor(shape, data)
	assert.NoErr(t, err)

	res := tensor.Softmax()
	assert.NotNil(t, res)
	assert.Equal(t, "Softmax", res.operator.name)
	assert.Equal(t, []*Tensor{tensor}, res.operands)
	err = res.Forward()
	assert.NoErr(t, err)
	assert.Equal(t, []float32{0.5, 0.5, 0.5, 0.5}, res.Data())
}

func TestTensorFormat(t *testing.T) {
	a, err := MakeTensor(Shape{2, 2}, []float32{1, 2, 3, 4})
	assert.NoErr(t, err)
	assert.NotNil(t, a)
	b, err := MakeTensor(Shape{3, 2}, []float32{5, 6, 7, 8, 9, 10})
	assert.NoErr(t, err)
	assert.NotNil(t, b)
	bias, err := MakeTensor(Shape{3}, []float32{1, 1, 1})
	assert.NoErr(t, err)
	assert.NotNil(t, bias)

	c := a.Matmul(b, bias)
	assert.NotNil(t, c)
	d, err := MakeTensor(Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	assert.NoErr(t, err)
	assert.NotNil(t, d)
	e := c.Matmul(d, nil)
	assert.NotNil(t, e)
	f := e.Softmax()
	assert.NotNil(t, f)
	err = f.Forward()
	assert.NoErr(t, err)
	assert.Equal(t, []float32{0, 1, 0, 1}, f.Data())
	graph := "Tensor (Softmax)\n    └── Tensor (Matmul)\n        ├── Tensor (Matmul)\n        │   ├── Tensor (2, 2)\n        │   ├── Tensor (3, 2)\n        │   └── Tensor (3)\n        ├── Tensor (2, 3)\n"
	// fmt.Printf("%g", f) // uncomment to see the tensor graph
	assert.Equal(t, graph, fmt.Sprintf("%g", f))
}

func TestTensorEmbedding(t *testing.T) {
	e, err := MakeTensor(Shape{3, 4}, []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	assert.NoErr(t, err)
	assert.NotNil(t, e)
	ids, err := MakeTensor(Shape{3}, []int32{1, 2, 0})
	assert.NoErr(t, err)
	assert.NotNil(t, ids)
	res := e.Embedding(ids)
	err = res.Forward()
	assert.NoErr(t, err)
	assert.Equal(t, []int32{5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4}, res.Data())
}
