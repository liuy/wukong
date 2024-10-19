package Array

import (
	"fmt"
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
