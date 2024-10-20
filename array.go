package Array

/*
#cgo LDFLAGS: ./build/libwukong.a -L/usr/local/cuda/lib64 -lcudnn -lcublasLt -lcudart

void cuda_init(void);
void cuda_fini(void);
void matmul(float *out, const float *inp, const float *weight, const float *bias,
			int batch, int row, int column, int oc);
void softmax(float* input, float* output, int row, int col);
*/
import "C"

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

type Runner interface {
	Softmax(a *Array) (*Array, error)
}

// Array is a multi-dimensional array of any type in a row-major order. It is represented by a Shape in the
// form of a slice of integers, e.g, [2, 3, 4] for a 3D Array and a storage that contains the data of any type
// in a contiguous block of memory.
type Array struct {
	Shape
	Storage
	Runner
}

// Returns the internal product of an int slice
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

// Returns the number of elements expected in a Array of a certain Shape
func (s Shape) Size() int {
	return product([]int(s))
}

// Returns the number of dimensions in the Shape
func (s Shape) Dims() int { return len(s) }

// Returns true if Array is scalar, false otherwise
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
		&cudaRunner{},
	}
}

// Creates a new Array from a Shape and a slice of data of any type
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

// Returns the element at a certain index in the Array
func (t *Array) GetElem(i int) any {
	return reflect.NewAt(t.dtype, unsafe.Pointer(&t.bytes[i*int(t.dtype.Size())])).Elem().Interface()
}

// Returns a slice of random float32 numbers of a certain size
func RandFloatSlice(size int) any {
	slice := make([]float32, size)
	for i := 0; i < size; i++ {
		slice[i] = rand.Float32()
	}
	return slice
}

func CudaSetup()                          { C.cuda_init() }
func CudaTeardown()                       { C.cuda_fini() }
func (a *Array) Softmax() (*Array, error) { return a.Runner.Softmax(a) }

// Run array operations on the CUDA device
type cudaRunner struct{}

func (r *cudaRunner) Softmax(a *Array) (*Array, error) {
	if a.Dims() < 2 {
		return nil, fmt.Errorf("Array must have at least 2 dimensions")
	}
	col := a.Shape[len(a.Shape)-1]
	row := a.Size() / col
	out := make([]byte, a.Size()*int(a.dtype.Size()))
	C.softmax((*C.float)(unsafe.Pointer(&a.bytes[0])), (*C.float)(unsafe.Pointer(&out[0])), C.int(row), C.int(col))
	return &Array{
		a.Shape,
		Storage{
			bytes: out,
			dtype: a.dtype,
		},
		a.Runner,
	}, nil
}
