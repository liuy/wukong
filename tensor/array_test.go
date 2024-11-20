package tensor

import (
	"fmt"
	"testing"

	"github.com/liuy/wukong/assert"
)

func TestMain(m *testing.M) {
	CudaSetup()
	m.Run()
	CudaTeardown()
}

func TestShapeLen(t *testing.T) {
	s := Shape{2, 3, 4}
	expected := 24
	assert.Equal(t, s.Len(), expected)
}

func TestArrayElemSize(t *testing.T) {
	tests := []struct {
		shape    Shape
		data     any
		expected int
	}{
		{Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6}, 4},
		{Shape{2, 3}, []int8{1, 2, 3, 4, 5, 6}, 1},
	}

	for _, tt := range tests {
		a, err := MakeArrayFrom(tt.shape, tt.data)
		assert.NoErr(t, err)
		assert.NotNil(t, a)
		assert.Equal(t, a.ElemSize(), tt.expected)
	}
}

func TestArraySize(t *testing.T) {
	tests := []struct {
		shape    Shape
		data     any
		expected int
	}{
		{Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6}, 24},
		{Shape{2, 3}, []int16{1, 2, 3, 4, 5, 6}, 12},
	}

	for _, tt := range tests {
		a, err := MakeArrayFrom(tt.shape, tt.data)
		assert.NoErr(t, err)
		assert.Equal(t, a.Size(), tt.expected)
	}
}

func TestShapeDims(t *testing.T) {
	s := Shape{2, 3, 4}
	expected := 3
	assert.Equal(t, s.Dims(), expected)
}

func TestShapeIsScalar(t *testing.T) {
	s := Shape{}
	assert.True(t, s.IsScalar())

	s = Shape{1}
	assert.False(t, s.IsScalar())
}

func TestArrayFormat(t *testing.T) {
	tests := []struct {
		shape    Shape
		data     any
		expected string
	}{
		{
			Shape{2, 2, 3},
			[]int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			"Shape: (2, 2, 3)\nType: int16\nData:\n 1 2 3\n 4 5 6\n\n 7 8 9\n 10 11 12\n",
		},
		{
			Shape{2, 3},
			[]float32{1.1, 2.2, 3.3, 4.4, 5.5, 6.6},
			"Shape: (2, 3)\nType: float32\nData:\n 1.1 2.2 3.3\n 4.4 5.5 6.6\n",
		},
	}

	for _, tt := range tests {
		a, err := MakeArrayFrom(tt.shape, tt.data)
		assert.NoErr(t, err)
		result := fmt.Sprintf("%v", a)
		assert.Equal(t, result, tt.expected)
	}
	// runtime.GC() // test if finalizer is called
}

func TestShapeFormat(t *testing.T) {
	s := Shape{2, 3, 4}
	expected := "(2, 3, 4)"
	result := fmt.Sprintf("%v", s)
	assert.Equal(t, result, expected)
}

func TestMakeArray(t *testing.T) {
	tests := []struct {
		Shape Shape
		data  any
		valid bool
	}{
		{Shape{4, 2, 2}, RandFloatSlice(16), true},
		{Shape{2, 3}, RandFloatSlice(6), true},
		{Shape{2, 2}, []int{1, 2, 3}, false},
	}

	for _, tt := range tests {
		_, err := MakeArrayFrom(tt.Shape, tt.data)
		assert.Equal(t, err == nil, tt.valid)
	}
}

func TestArraySoftmax(t *testing.T) {
	tests := []struct {
		shape    Shape
		input    []float32
		expected []float32
	}{
		{Shape{1, 3}, []float32{1.0, 1.0, 1.0}, []float32{0.33333333, 0.33333333, 0.33333333}},
		{Shape{2, 1, 3}, []float32{3.0, 1.0, 0.2, 1, 1000, 2}, []float32{0.8360188, 0.11314284, 0.05083836, 0.0, 1.0, 0.0}},
	}

	a, err := MakeArrayFrom(Shape{3}, []float32{1.0, 2.0, 3.0})
	assert.NoErr(t, err)
	res, err := a.Softmax()
	assert.NoErr(t, err)
	assert.SliceNear(t, []float32{0.0900305, 0.2447284, 0.6652409}, res.ToHost().Interface().([]float32), 1e-6)

	for _, tt := range tests {
		array, err := MakeArrayFrom(tt.shape, tt.input)
		assert.NoErr(t, err)

		res, err := array.Softmax()
		assert.NoErr(t, err)
		assert.SliceNear(t, tt.expected, res.ToHost().Interface().([]float32), 1e-6)
	}
}

func TestArrayMatmul(t *testing.T) {
	tests := []struct {
		a, b, bias                       Shape
		aData, bData, biasData, expected []float32
		expectError                      bool
	}{
		{
			a: Shape{2, 3}, b: Shape{2, 3}, bias: Shape{2},
			aData: []float32{1, 2, 3, 4, 5, 6}, bData: []float32{7, 9, 11, 8, 10, 12}, biasData: []float32{1, 1},
			expected: []float32{59, 65, 140, 155}, expectError: false,
		},
		{
			a: Shape{2, 3}, b: Shape{2, 3}, bias: Shape{},
			aData: []float32{1, 2, 3, 4, 5, 6}, bData: []float32{7, 9, 11, 8, 10, 12}, biasData: nil,
			expected: []float32{58, 64, 139, 154}, expectError: false,
		},
		{
			a: Shape{2, 2, 3}, b: Shape{2, 3}, bias: Shape{2},
			aData: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, bData: []float32{13, 15, 17, 14, 16, 18}, biasData: []float32{1, 1},
			expected: []float32{95, 101, 230, 245, 365, 389, 500, 533}, expectError: false,
		},
		{
			a: Shape{2, 3}, b: Shape{2, 2}, bias: Shape{},
			aData: []float32{1, 2, 3, 4, 5, 6}, bData: []float32{7, 8, 9, 10}, biasData: nil,
			expected: nil, expectError: true,
		},
	}

	for _, tt := range tests {
		a, err := MakeArrayFrom(tt.a, tt.aData)
		assert.NoErr(t, err)
		b, err := MakeArrayFrom(tt.b, tt.bData)
		assert.NoErr(t, err)
		var bias *Array
		if tt.biasData != nil {
			bias, err = MakeArrayFrom(tt.bias, tt.biasData)
			assert.NoErr(t, err)
		}

		result, err := a.Matmul(b, bias)
		assert.Equal(t, err != nil, tt.expectError)
		if err != nil {
			continue
		}
		assert.SliceNear(t, tt.expected, result.ToHost().Interface().([]float32), 1e-6)
	}
}

func TestArrayMatmulSoftmax(t *testing.T) {
	a, err := MakeArrayFrom(Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	assert.NoErr(t, err)
	b, err := MakeArrayFrom(Shape{3, 3}, []float32{1, 4, 7, 2, 5, 8, 3, 6, 9})
	assert.NoErr(t, err)
	c, err := a.Matmul(b, nil)
	assert.NoErr(t, err)
	d, err := c.Softmax()
	assert.NoErr(t, err)
	expected := []float32{6.1289825e-06, 0.0024726081, 0.9975212, 9.3576195e-14, 3.059022e-07, 0.99999964}
	assert.SliceNear(t, expected, d.ToHost().Interface().([]float32), 1e-6)
}
