package Array

import (
	"fmt"
	"math/rand"
	"reflect"
	"runtime"
	"unsafe"
)

type Shape []int

type Storage struct {
	bytes []byte
	dtype reflect.Type
}

// Array is a multi-dimensional array of any type in a row-major order. It is represented by a Shape in the
// form of a slice of integers, e.g, [2, 3, 4] for a 3D Array and a storage that contains the data of any type
// in a contiguous block of memory.
type Array struct {
	Shape
	Storage
}

// returns the internal product of an int slice
func product(a []int) (ret int) {
	ret = 1
	if len(a) == 0 {
		return
	}
	for _, v := range a {
		ret *= v
	}
	return
}

// returns the number of elements expected in a Array of a certain Shape
func (s Shape) Size() int {
	return product([]int(s))
}

// returns the number of dimensions in the Shape
func (s Shape) Dims() int { return len(s) }

// returns true if Array is scalar, false otherwise
func (s Shape) IsScalar() bool { return s.Dims() == 0 }

func (s Shape) Format(st fmt.State, r rune) {
	switch r {
	case 'v', 's':
		st.Write([]byte("("))
		for i, v := range s {
			fmt.Fprintf(st, "%d", v)
			if i < len(s)-1 {
				st.Write([]byte(", "))
			}
		}
		st.Write([]byte(")"))
	default:
		fmt.Fprintf(st, "%v", []int(s))
	}
}

func MakeArray(s Shape, t reflect.Type) *Array {
	return &Array{
		s,
		Storage{
			bytes: make([]byte, s.Size()*int(t.Size())),
			dtype: t,
		},
	}
}

// creates a new Array from a Shape and a slice of data of any type
// For e.g, MakeArrayFrom(Shape{2, 3}, []int8{1, 2, 3, 4, 5, 6}) creates a 2D Array of int8
func MakeArrayFrom(s Shape, data any) (ret *Array, e error) {
	v := reflect.ValueOf(data)
	if v.Len() != s.Size() {
		_, file, line, ok := runtime.Caller(1)
		if !ok {
			return nil, fmt.Errorf("data length does not match Shape")
		}
		return nil, fmt.Errorf("data length does not match Shape (file: %s, line: %d)", file, line)
	}

	t := reflect.TypeOf(v.Index(0).Interface())
	ret = MakeArray(s, t)

	for i := 0; i < v.Len(); i++ {
		val := v.Index(i)
		ptr := unsafe.Pointer(&ret.bytes[i*int(t.Size())])
		reflect.NewAt(t, ptr).Elem().Set(val)
	}

	return ret, nil
}

func (t *Array) Format(st fmt.State, r rune) {
	s := fmt.Sprintf("Shape: %v\nType: %v", t.Shape, t.dtype)

	data := "\nData:\n"
	dims := t.Shape
	stride := dims[len(dims)-1]

	for i := 0; i < t.Size(); i++ {
		if i > 0 && i%stride == 0 {
			data += "\n"
			if i%(stride*dims[len(dims)-2]) == 0 {
				data += "\n"
			}
		}
		data += fmt.Sprintf(" %v", reflect.NewAt(t.dtype, unsafe.Pointer(&t.bytes[i*int(t.dtype.Size())])).Elem())
	}
	data += "\n"

	fmt.Fprintf(st, "%s%s", s, data)
}

// returns a slice of random float32 numbers of a certain size
func RandFloatSlice(size int) any {
	slice := make([]float32, size)
	for i := 0; i < size; i++ {
		slice[i] = rand.Float32()
	}
	return slice
}
