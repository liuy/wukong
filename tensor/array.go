package tensor

/*
#cgo LDFLAGS: ./build/libwukong.a -L/usr/local/cuda/lib64 -lcudnn -lcublasLt -lcudart -lm -lcublas

#include "../cuda.h"
*/
import "C"

import (
	"fmt"
	"math/rand"
	"reflect"
	"runtime"
	"unsafe"
)

// DType represents the type of ondisk tensor data, extended from GGMLType
type DType uint32

const (
	GGML_TYPE_F32      DType = iota // 0
	GGML_TYPE_F16                   // 1
	GGML_TYPE_Q4_0                  // 2
	GGML_TYPE_Q4_1                  // 3
	GGML_TYPE_Q4_2                  // 4 support has been removed
	GGML_TYPE_Q4_3                  // 5 support has been removed
	GGML_TYPE_Q5_0                  // 6
	GGML_TYPE_Q5_1                  // 7
	GGML_TYPE_Q8_0                  // 8
	GGML_TYPE_Q8_1                  // 9
	GGML_TYPE_Q2_K                  // 10
	GGML_TYPE_Q3_K                  // 11
	GGML_TYPE_Q4_K                  // 12
	GGML_TYPE_Q5_K                  // 13
	GGML_TYPE_Q6_K                  // 14
	GGML_TYPE_Q8_K                  // 15
	GGML_TYPE_IQ2_XXS               // 16
	GGML_TYPE_IQ2_XS                // 17
	GGML_TYPE_IQ3_XXS               // 18
	GGML_TYPE_IQ1_S                 // 19
	GGML_TYPE_IQ4_NL                // 20
	GGML_TYPE_IQ3_S                 // 21
	GGML_TYPE_IQ2_S                 // 22
	GGML_TYPE_IQ4_XS                // 23
	GGML_TYPE_I8                    // 24
	GGML_TYPE_I16                   // 25
	GGML_TYPE_I32                   // 26
	GGML_TYPE_I64                   // 27
	GGML_TYPE_F64                   // 28
	GGML_TYPE_IQ1_M                 // 29
	GGML_TYPE_BF16                  // 30
	GGML_TYPE_Q4_0_4_4              // 31
	GGML_TYPE_Q4_0_4_8              // 32
	GGML_TYPE_Q4_0_8_8              // 33
	GGML_TYPE_TQ1_0                 // 34
	GGML_TYPE_TQ2_0                 // 35
)

type Shape []int

type Storage struct {
	dptr  unsafe.Pointer
	atype reflect.Type // Type of the Array to interact with go
	dtype DType
}

type Runner interface {
	Softmax(a *Array) (*Array, error)
	Matmul(a, b, bias *Array) (*Array, error)
	ToDevice(a *Array, src unsafe.Pointer)
	ToHost(a *Array) reflect.Value
	DeviceFree(a *Array)
	Embedding(embd, ids *Array) (*Array, error)
}

// Array is a multi-dimensional array of any type in a row-major order. It is represented by a Shape in the
// form of a slice of integers, e.g, [2, 3, 4] for a 3D Array and a storage that contains the data of any type
// in a contiguous block of memory, which a runner can perform device specific operations on.
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

// Returns the number of elements expected in the Array
func (s Shape) Len() int {
	return product([]int(s))
}

// NumDims Returns the number of dimensions in the Shape
func (s Shape) NumDims() int { return len(s) }

// DimAt returns the dimension at a given index with support for negative indexing.
// Out of range indices will panic
func (s Shape) DimAt(idx int) int {
	if idx < 0 {
		return s[len(s)+idx]
	}
	return s[idx]
}

// Returns true if Array is scalar, false otherwise
func (s Shape) IsScalar() bool { return s.NumDims() == 0 }

func (s Shape) Format(st fmt.State, r rune) {
	st.Write([]byte("("))
	for i, v := range s {
		fmt.Fprintf(st, "%d", v)
		if i < len(s)-1 {
			st.Write([]byte(", "))
		}
	}
	st.Write([]byte(")"))
}

func NewArray(s Shape, t reflect.Type) *Array {
	a := &Array{
		s,
		Storage{
			atype: t,
		},
		&cudaRunner{},
	}
	runtime.SetFinalizer(a, (*Array).DeviceFree)
	return a
}

// Creates a new Array from a Shape and a slice of data of any type
// For e.g, MakeArrayFrom(Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6}) creates a 2D Array of float32
func MakeArrayFrom(s Shape, data any) (ret *Array, e error) {
	v := reflect.ValueOf(data)
	if v.Len() != s.Len() {
		_, file, line, ok := runtime.Caller(1)
		if !ok {
			return nil, fmt.Errorf("data length does not match Shape")
		}
		return nil, fmt.Errorf("data length does not match Shape (file: %s, line: %d)", file, line)
	}
	ret = NewArray(s, v.Type())
	ret.ToDevice(unsafe.Pointer(v.Pointer()))
	return ret, nil
}

func (t *Array) Format(st fmt.State, r rune) {
	s := fmt.Sprintf("Shape: %v\nType: %v", t.Shape, t.ElemType())
	data := "\nData:\n"
	stride := t.DimAt(-1)
	d := t.ToHost()
	for i := 0; i < t.Len(); i++ {
		if i > 0 && i%stride == 0 {
			data += "\n"
			if i%(stride*t.DimAt(-2)) == 0 {
				data += "\n"
			}
		}
		data += fmt.Sprintf(" %v", d.Index(i))
	}
	data += "\n"
	fmt.Fprintf(st, "%s%s", s, data)
}

// Returns a slice of random float32 numbers of a certain size
func RandFloatSlice(size int) any {
	slice := make([]float32, size)
	for i := 0; i < size; i++ {
		slice[i] = rand.Float32()
	}
	return slice
}

func CudaSetup()    { C.cuda_init() }
func CudaTeardown() { C.cuda_fini() }

// Returns the element type of the Array
func (a *Array) ElemType() reflect.Type { return a.atype.Elem() }

// Returns the element size of the Array in bytes
func (a *Array) ElemSize() int { return int(a.atype.Elem().Size()) }

// Returns the total size of the Array in bytes
func (a *Array) Size() int { return a.Len() * a.ElemSize() }

// Softmax in a row-wise manner
func (a *Array) Softmax() (*Array, error) { return a.Runner.Softmax(a) }

// Fused matrix multiplication: a @ b + bias(could be nil).
func (a *Array) Matmul(b, bias *Array) (*Array, error) { return a.Runner.Matmul(a, b, bias) }

// Copy data from host to device
func (a *Array) ToDevice(src unsafe.Pointer) { a.Runner.ToDevice(a, src) }

// Copy data from device to host
func (a *Array) ToHost() reflect.Value { return a.Runner.ToHost(a) }

// Free device memory
func (a *Array) DeviceFree() { a.Runner.DeviceFree(a) }

// Embedding returns the embeddings of the given idsss
func (a *Array) Embedding(ids *Array) (*Array, error) { return a.Runner.Embedding(a, ids) }

// Run array operations on the CUDA device
type cudaRunner struct{}

func (r *cudaRunner) ToDevice(a *Array, src unsafe.Pointer) {
	if a.dptr == nil {
		a.dptr = C.cuda_malloc(C.size_t(a.Size()))
	}
	C.cuda_to_device(a.dptr, src, C.size_t(a.Size()))
}

func (r *cudaRunner) ToHost(a *Array) reflect.Value {
	dst := reflect.MakeSlice(a.atype, a.Len(), a.Len())
	C.cuda_to_host(unsafe.Pointer(dst.Pointer()), a.dptr, C.size_t(a.Size()))
	return dst
}

func (r *cudaRunner) DeviceFree(a *Array) {
	if a.dptr != nil {
		// fmt.Printf("Freeing device memory at %p\n", a.dptr)
		C.cuda_free(a.dptr)
		a.dptr = nil
	}
}

func (r *cudaRunner) Softmax(a *Array) (*Array, error) {
	col := a.DimAt(-1)
	row := a.Len() / col
	out := C.cuda_malloc(C.size_t(row * col * a.ElemSize()))
	C.cuda_softmax(out, a.dptr, C.int(row), C.int(col))
	ret := NewArray(a.Shape, a.atype)
	ret.dptr = out
	return ret, nil
}

func (r *cudaRunner) Matmul(a, b, bias *Array) (*Array, error) {
	// For now we only support b as a 2D array
	if a.NumDims() < 2 || b.NumDims() != 2 {
		return nil, fmt.Errorf("arrays must have at least 2 dimensions")
	}
	if a.DimAt(-1) != b.DimAt(-1) {
		return nil, fmt.Errorf("array shapes do not match")
	}
	var biasPtr unsafe.Pointer = nil
	if bias != nil {
		biasPtr = bias.dptr
		if bias.Len() != b.DimAt(-2) {
			return nil, fmt.Errorf("bias shape does not match")
		}
	}
	column := a.DimAt(-1)
	row := a.Len() / column
	oc := b.DimAt(-2)
	out := C.cuda_malloc(C.size_t(row * oc * a.ElemSize()))
	C.cuda_matmul(out, a.dptr, b.dptr, biasPtr, C.int(row), C.int(column), C.int(oc))
	shape := a.Shape
	shape[len(shape)-1] = oc
	ret := NewArray(shape, a.atype)
	ret.dptr = out
	return ret, nil
}

func (r cudaRunner) Embedding(embd, ids *Array) (*Array, error) {
	if ids.ElemType() != reflect.TypeOf(int32(0)) {
		return nil, fmt.Errorf("index must be an int32 integer")
	}
	if ids.NumDims() > 2 {
		return nil, fmt.Errorf("index must be 1D or 2D")
	}
	if embd.NumDims() != 2 {
		return nil, fmt.Errorf("embedding must be 2D")
	}
	col := embd.DimAt(1)
	row := ids.DimAt(-1)
	batch := 1
	if ids.NumDims() == 2 {
		batch = ids.DimAt(0)
	}
	out := C.cuda_malloc(C.size_t(batch * row * col * embd.ElemSize()))
	C.cuda_embedding(out, ids.dptr, embd.dptr, C.int(batch), C.int(row), C.int(col))
	ret := NewArray(Shape{batch, row, col}, embd.atype)
	ret.dptr = out
	return ret, nil
}
