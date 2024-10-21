package Array

import (
	"fmt"
	"math"
	"testing"
)

func TestShapeSize(t *testing.T) {
	s := Shape{2, 3, 4}
	expected := 24
	if s.Size() != expected {
		t.Errorf("Shape.size() = %d, want %d", s.Size(), expected)
	}
}

func TestShapeDims(t *testing.T) {
	s := Shape{2, 3, 4}
	expected := 3
	if s.Dims() != expected {
		t.Errorf("Shape.dims() = %d, want %d", s.Dims(), expected)
	}
}

func TestShapeIsScalar(t *testing.T) {
	s := Shape{}
	if !s.IsScalar() {
		t.Errorf("Shape.IsScalar() = false, want true")
	}
	s = Shape{1}
	if s.IsScalar() {
		t.Errorf("Shape.IsScalar() = true, want false")
	}
}

func TestArrayFormat(t *testing.T) {
	Array, err := MakeArrayFrom(Shape{2, 2, 3}, []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	if err != nil {
		t.Fatalf("MakeArrayFrom() error = %v", err)
	}
	expected := "Shape: (2, 2, 3)\nType: int8\nData:\n 1 2 3\n 4 5 6\n\n 7 8 9\n 10 11 12\n"
	result := fmt.Sprintf("%v", Array)
	if result != expected {
		t.Errorf("Array.Format() = \n%v\nWant \n%v", result, expected)
	}
}

func TestShapeFormat(t *testing.T) {
	s := Shape{2, 3, 4}
	expected := "(2, 3, 4)"
	result := fmt.Sprintf("%v", s)
	if result != expected {
		t.Errorf("Shape.Format() = %v, want %v", result, expected)
	}
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
		if (err == nil) != tt.valid {
			t.Errorf("MakeArrayFrom(%v, %v) = %v, want valid = %v", tt.Shape, tt.data, err, tt.valid)
		}
	}
}

func TestArraySoftmax(t *testing.T) {
	CudaSetup()
	defer CudaTeardown()

	tests := []struct {
		shape    Shape
		input    []float32
		expected []float32
	}{
		{Shape{1, 3}, []float32{1.0, 1.0, 1.0}, []float32{0.33333333, 0.33333333, 0.33333333}},
		{Shape{2, 1, 3}, []float32{3.0, 1.0, 0.2, 1, 1000, 2}, []float32{0.8360188, 0.11314284, 0.05083836, 0.0, 1.0, 0.0}},
	}

	a, err := MakeArrayFrom(Shape{3}, []float32{1.0, 2.0, 3.0})
	if err != nil {
		t.Fatalf("MakeArrayFrom() error = %v", err)
	}
	if _, err := a.Softmax(); err == nil {
		t.Fatal("Array.Softmax() should have failed")
	}

	for _, tt := range tests {
		array, err := MakeArrayFrom(tt.shape, tt.input)
		if err != nil {
			t.Fatalf("MakeArrayFrom(%v, %v) failed: %v", tt.shape, tt.input, err)
		}

		res, err := array.Softmax()
		if err != nil {
			t.Fatalf("Array.Softmax() failed: %v", err)
		}
		if !equal(res, tt.expected) {
			t.Errorf("Array.Softmax() = %v, want %v", res, tt.expected)
		}
	}
}

func equal(a *Array, expected []float32) bool {
	if a.Size() != len(expected) {
		return false
	}

	for i, v := range expected {
		if math.Abs(float64(a.GetElem(i).(float32)-v)) > 1e-6 {
			return false
		}
	}
	return true
}

func TestArrayMatmul(t *testing.T) {
	CudaSetup()
	defer CudaTeardown()

	tests := []struct {
		a, b, bias                       Shape
		aData, bData, biasData, expected []float32
		expectError                      bool
	}{
		{
			a: Shape{2, 3}, b: Shape{3, 2}, bias: Shape{2},
			aData: []float32{1, 2, 3, 4, 5, 6}, bData: []float32{7, 8, 9, 10, 11, 12}, biasData: []float32{1, 1},
			expected: []float32{59, 65, 140, 155}, expectError: false,
		},
		{
			a: Shape{2, 3}, b: Shape{3, 2}, bias: Shape{},
			aData: []float32{1, 2, 3, 4, 5, 6}, bData: []float32{7, 8, 9, 10, 11, 12}, biasData: nil,
			expected: []float32{58, 64, 139, 154}, expectError: false,
		},
		{
			a: Shape{2, 2, 3}, b: Shape{3, 2}, bias: Shape{2},
			aData: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, bData: []float32{13, 14, 15, 16, 17, 18}, biasData: []float32{1, 1},
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
		if err != nil {
			t.Fatalf("MakeArrayFrom(%v, %v) failed: %v", tt.a, tt.aData, err)
		}
		b, err := MakeArrayFrom(tt.b, tt.bData)
		if err != nil {
			t.Fatalf("MakeArrayFrom(%v, %v) failed: %v", tt.b, tt.bData, err)
		}
		var bias *Array
		if tt.biasData != nil {
			bias, err = MakeArrayFrom(tt.bias, tt.biasData)
			if err != nil {
				t.Fatalf("MakeArrayFrom(%v, %v) failed: %v", tt.bias, tt.biasData, err)
			}
		}

		result, err := a.Matmul(b, bias)
		if (err != nil) != tt.expectError {
			t.Errorf("Array.Matmul() error = %v, expectError %v", err, tt.expectError)
			continue
		}
		if err != nil {
			continue
		}

		if !equal(result, tt.expected) {
			t.Errorf("Array.Matmul() = %v, want %v", result, tt.expected)
		}
	}
}
