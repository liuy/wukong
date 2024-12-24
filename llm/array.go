package llm

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
	GGML_TYPE_COUNT
)

const QK_K = 256

// QuantInfo holds information about a quantization type
type DTypeSize struct {
	name      string
	blockSize int
	typeSize  int
}

// DTypeInfo maps DType to quantization information {name, blockSize, typeSize}
var DTypeInfo = map[DType]DTypeSize{
	GGML_TYPE_F32:      {"F32", 1, 4},
	GGML_TYPE_F16:      {"F16", 1, 2},
	GGML_TYPE_Q4_0:     {"GGML_TYPE_Q4_0", 32, 2 + 16},
	GGML_TYPE_Q4_1:     {"GGML_TYPE_Q4_1", 32, 2 + 2 + 16},
	GGML_TYPE_Q4_2:     {"GGML_TYPE_Q4_2", 32, 2 + 2 + 16},
	GGML_TYPE_Q4_3:     {"GGML_TYPE_Q4_3", 32, 2 + 2 + 16},
	GGML_TYPE_Q5_0:     {"GGML_TYPE_Q5_0", 32, 2 + 4 + 16},
	GGML_TYPE_Q5_1:     {"GGML_TYPE_Q5_1", 32, 2 + 2 + 4 + 16},
	GGML_TYPE_Q8_0:     {"GGML_TYPE_Q8_0", 32, 2 + 32},
	GGML_TYPE_Q8_1:     {"GGML_TYPE_Q8_1", 32, 4 + 4 + 32},
	GGML_TYPE_Q2_K:     {"GGML_TYPE_Q2_K", 256, 2 + 2 + QK_K/16 + QK_K/4},
	GGML_TYPE_Q3_K:     {"GGML_TYPE_Q3_K", 256, 2 + QK_K/4 + QK_K/8 + 12},
	GGML_TYPE_Q4_K:     {"GGML_TYPE_Q4_K", 256, 2 + 2 + QK_K/2 + 12},
	GGML_TYPE_Q5_K:     {"GGML_TYPE_Q5_K", 256, 2 + 2 + QK_K/2 + QK_K/8 + 12},
	GGML_TYPE_Q6_K:     {"GGML_TYPE_Q6_K", 256, 2 + QK_K/2 + QK_K/4 + QK_K/16},
	GGML_TYPE_Q8_K:     {"GGML_TYPE_Q8_K", 256, 4 + QK_K + QK_K/8},
	GGML_TYPE_IQ2_XXS:  {"GGML_TYPE_IQ2_XXS", 256, 2 + QK_K/4},
	GGML_TYPE_IQ2_XS:   {"GGML_TYPE_IQ2_XS", 256, 2 + QK_K/4 + QK_K/32},
	GGML_TYPE_IQ3_XXS:  {"GGML_TYPE_IQ3_XXS", 256, 2 + QK_K/4 + QK_K/8},
	GGML_TYPE_IQ1_S:    {"GGML_TYPE_IQ1_S", 256, 2 + QK_K/8 + QK_K/16},
	GGML_TYPE_IQ4_NL:   {"GGML_TYPE_IQ4_NL", 32, 2 + 16},
	GGML_TYPE_IQ3_S:    {"GGML_TYPE_IQ3_S", 256, 2 + QK_K/4 + QK_K/8 + QK_K/32 + 4},
	GGML_TYPE_IQ2_S:    {"GGML_TYPE_IQ2_S", 256, 2 + QK_K/4 + QK_K/16},
	GGML_TYPE_IQ4_XS:   {"GGML_TYPE_IQ4_XS", 256, 2 + 2 + QK_K/2 + QK_K/64},
	GGML_TYPE_I8:       {"Int8", 1, 1},
	GGML_TYPE_I16:      {"Int16", 1, 2},
	GGML_TYPE_I32:      {"Int32", 1, 4},
	GGML_TYPE_I64:      {"Int64", 1, 8},
	GGML_TYPE_F64:      {"F64", 1, 8},
	GGML_TYPE_IQ1_M:    {"GGML_TYPE_IQ1_M", 256, QK_K/8 + QK_K/16 + QK_K/32},
	GGML_TYPE_BF16:     {"BF16", 1, 2},
	GGML_TYPE_Q4_0_4_4: {"GGML_TYPE_Q4_0_4_4", 32, 2 + 16},
	GGML_TYPE_Q4_0_4_8: {"GGML_TYPE_Q4_0_4_8", 32, 2 + 16},
	GGML_TYPE_Q4_0_8_8: {"GGML_TYPE_Q4_0_8_8", 32, 2 + 16},
	GGML_TYPE_TQ1_0:    {"GGML_TYPE_TQ1_0", 256, 2 + 4*13},
	GGML_TYPE_TQ2_0:    {"GGML_TYPE_TQ2_0", 256, 2 + 64},
}

func (t DType) String() string {
	if info, ok := DTypeInfo[t]; ok {
		return info.name
	}
	return fmt.Sprintf("GGML_TYPE_UNKNOWN(%d)", t)
}

type Shape []int

type Storage struct {
	dptr  unsafe.Pointer
	dtype DType
}

type Runner interface {
	Softmax(a *Array) (*Array, error)
	Matmul(a, b, bias *Array) (*Array, error)
	ToDevice(a *Array, src unsafe.Pointer)
	ToHost(a *Array) any
	DeviceFree(a *Array)
	Embedding(embd, ids *Array) (*Array, error)
	Rmsnorm(x, w *Array, eps float32) (*Array, error)
	Cat(a, b *Array) (*Array, error)
	DivInPlace(a, b *Array) error
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

// GetDim returns the dimension at a given index with support for negative indexing.
// Out of range indices will panic
func (s Shape) GetDim(idx int) int {
	if idx < 0 {
		return s[len(s)+idx]
	}
	return s[idx]
}

// Sets the dimension at a given index with support for negative indexing.
// Out of range indices will panic
func (s Shape) SetDim(idx int, v int) {
	if idx < 0 {
		s[len(s)+idx] = v
	} else {
		s[idx] = v
	}
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

func NewArray(s Shape, d DType) *Array {
	a := &Array{
		s,
		Storage{
			dtype: d,
		},
		&cudaRunner{},
	}
	runtime.SetFinalizer(a, (*Array).DeviceFree)
	return a
}

// Creates a new Array from a Shape and data of DType. For e.g, create a 2D Array of float32
//
//	data := []float32{1, 2, 3, 4, 5, 6}
//	ptr := unsafe.Pointer(&data[0])
//	MakeArrayFrom(Shape{2, 3}, ptr, GGML_TYPE_F32)
//
// Parameters:
//
//	s : shape of the data
//	p : pointer to the data in the memory
//	t : DType of the data
func MakeArrayFrom(s Shape, p unsafe.Pointer, t DType) (*Array, error) {
	if s.Len() <= 0 || s.NumDims() == 0 || s == nil {
		return nil, fmt.Errorf("bad shape: %v", s)
	}
	if p == nil {
		return nil, fmt.Errorf("bad pointer: %v", p)
	}
	info, ok := DTypeInfo[t]
	if !ok {
		return nil, fmt.Errorf("bad dtype: %v", t)
	}
	if s.GetDim(-1)%info.blockSize != 0 {
		return nil, fmt.Errorf("shape %v is not aligned to %d", s, info.blockSize)
	}
	a := NewArray(s, t)
	a.ToDevice(p)
	return a, nil
}

func DTypeOf(t reflect.Type) DType {
	switch t.Kind() {
	case reflect.Int8:
		return GGML_TYPE_I8
	case reflect.Int16:
		return GGML_TYPE_I16
	case reflect.Int32:
		return GGML_TYPE_I32
	case reflect.Int64:
		return GGML_TYPE_I64
	case reflect.Float32:
		return GGML_TYPE_F32
	case reflect.Float64:
		return GGML_TYPE_F64
	default:
		panic(fmt.Sprintf("unsupported type %v", t))
	}
}

// Creates a new Array from a Shape and a go slice of data of any type
// For e.g,
//
//	MakeArray(Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
//
// creates a 2D Array of float32
func MakeArray(s Shape, data any) (ret *Array, e error) {
	if s == nil {
		return nil, fmt.Errorf("shape is nil")
	}
	if data == nil {
		return nil, fmt.Errorf("data is nil")
	}
	v := reflect.ValueOf(data)
	if v.Len() != s.Len() {
		return nil, fmt.Errorf("data length %d does not match shape length %d", v.Len(), s.Len())
	}
	ret = NewArray(s, DTypeOf(v.Type().Elem()))
	ret.ToDevice(unsafe.Pointer(v.Pointer()))
	return ret, nil
}

func (t *Array) Format(st fmt.State, r rune) {
	s := fmt.Sprintf("Shape: %v\nType: %v", t.Shape, t.ElemType())
	data := "\nData:\n"
	stride := t.GetDim(-1)
	d := reflect.ValueOf(t.ToHost())
	for i := 0; i < t.Len(); i++ {
		if i > 0 && i%stride == 0 {
			data += "\n"
			if i%(stride*t.GetDim(-2)) == 0 {
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
func (a *Array) ElemType() DType { return a.dtype }

// Returns the element size of the Array in bytes
func (a *Array) ElemBlockSize() int {
	if info, ok := DTypeInfo[a.dtype]; ok {
		return info.blockSize
	}
	panic(fmt.Sprintf("Bad dtype: %d", a.dtype))
}

func (a *Array) ElemTypeSize() int {
	if info, ok := DTypeInfo[a.dtype]; ok {
		return info.typeSize
	}
	panic(fmt.Sprintf("Bad dtype: %d", a.dtype))
}

// Returns the total size of the Array in bytes
func (a *Array) Size() int {
	info, ok := DTypeInfo[a.dtype]
	if !ok {
		panic(fmt.Sprintf("Bad dtype: %d", a.dtype))
	}

	s := a.Shape
	if s.GetDim(-1)%info.blockSize != 0 {
		panic(fmt.Errorf("Shape %v is not aligned to %d", s, info.blockSize))
	}

	return a.Len() / info.blockSize * info.typeSize
}

// Softmax in a row-wise manner
func (a *Array) Softmax() (*Array, error) { return a.Runner.Softmax(a) }

// Fused matrix multiplication: a @ b + bias(could be nil).
func (a *Array) Matmul(b, bias *Array) (*Array, error) { return a.Runner.Matmul(a, b, bias) }

// Copy data from host to device
func (a *Array) ToDevice(src unsafe.Pointer) { a.Runner.ToDevice(a, src) }

// Copy data from device to host
func (a *Array) ToHost() any { return a.Runner.ToHost(a) }

// Free device memory
func (a *Array) DeviceFree() { a.Runner.DeviceFree(a) }

// Embedding returns the embeddings of the given idsss
func (a *Array) Embedding(ids *Array) (*Array, error) { return a.Runner.Embedding(a, ids) }

// Rmsnorm returns the root mean square normalization of the given array
func (a *Array) Rmsnorm(x *Array, eps float32) (*Array, error) { return a.Runner.Rmsnorm(a, x, eps) }

// Cat returns the concatenation of the given arrays along the first dimension
func (a *Array) Cat(b *Array) (*Array, error) { return a.Runner.Cat(a, b) }

// DivInPlace divides the array a by b in place
func (a *Array) DivInPlace(b *Array) error { return a.Runner.DivInPlace(a, b) }

// Run array operations on the CUDA device
type cudaRunner struct{}

func (r *cudaRunner) ToDevice(a *Array, src unsafe.Pointer) {
	if a.dptr == nil {
		a.dptr = C.cuda_malloc(C.size_t(a.Size()))
	}
	C.cuda_to_device(a.dptr, src, C.size_t(a.Size()))
}

func (r *cudaRunner) ToHost(a *Array) any {
	dst := make([]byte, a.Size())
	C.cuda_to_host(unsafe.Pointer(&dst[0]), a.dptr, C.size_t(a.Size()))
	switch a.dtype {
	case GGML_TYPE_F32:
		return unsafe.Slice((*float32)(unsafe.Pointer(&dst[0])), a.Len())
	case GGML_TYPE_F64:
		return unsafe.Slice((*float64)(unsafe.Pointer(&dst[0])), a.Len())
	case GGML_TYPE_I8:
		return unsafe.Slice((*int8)(unsafe.Pointer(&dst[0])), a.Len())
	case GGML_TYPE_I16:
		return unsafe.Slice((*int16)(unsafe.Pointer(&dst[0])), a.Len())
	case GGML_TYPE_I32:
		return unsafe.Slice((*int32)(unsafe.Pointer(&dst[0])), a.Len())
	case GGML_TYPE_I64:
		return unsafe.Slice((*int64)(unsafe.Pointer(&dst[0])), a.Len())
	default:
		panic(fmt.Sprintf("unsupported dtype %v", a.dtype))
	}
}

func (r *cudaRunner) DeviceFree(a *Array) {
	if a.dptr != nil {
		// fmt.Printf("Freeing device memory at %p\n", a.dptr)
		C.cuda_free(a.dptr)
		a.dptr = nil
	}
}

func (r *cudaRunner) Softmax(a *Array) (*Array, error) {
	col := a.GetDim(-1)
	row := a.Len() / col
	out := C.cuda_malloc(C.size_t(row * col * a.ElemTypeSize() / a.ElemBlockSize()))
	C.cuda_softmax(out, a.dptr, C.int(row), C.int(col))
	ret := NewArray(a.Shape, a.dtype)
	ret.dptr = out
	return ret, nil
}

func (r *cudaRunner) Matmul(a, b, bias *Array) (*Array, error) {
	// For now we only support b as a 2D array
	if a.NumDims() < 2 || b.NumDims() != 2 {
		return nil, fmt.Errorf("arrays must have at least 2 dimensions")
	}
	if a.GetDim(-1) != b.GetDim(-1) {
		return nil, fmt.Errorf("array shapes do not match")
	}
	var biasPtr unsafe.Pointer = nil
	if bias != nil {
		biasPtr = bias.dptr
		if bias.Len() != b.GetDim(-2) {
			return nil, fmt.Errorf("bias shape does not match")
		}
	}
	column := a.GetDim(-1)
	row := a.Len() / column
	oc := b.GetDim(-2)
	out := C.cuda_malloc(C.size_t(row * oc * a.ElemTypeSize() / a.ElemBlockSize()))
	C.cuda_matmul(out, a.dptr, b.dptr, biasPtr, C.int(row), C.int(column), C.int(oc))
	shape := a.Shape
	shape[len(shape)-1] = oc
	ret := NewArray(shape, a.dtype)
	ret.dptr = out
	return ret, nil
}

func (r *cudaRunner) Embedding(embd, ids *Array) (*Array, error) {
	if ids.ElemType() != GGML_TYPE_I32 {
		return nil, fmt.Errorf("index must be an int32 integer")
	}
	if ids.NumDims() > 2 {
		return nil, fmt.Errorf("index must be 1D or 2D")
	}
	if embd.NumDims() != 2 {
		return nil, fmt.Errorf("embedding must be 2D")
	}
	col := embd.GetDim(1)
	row := ids.GetDim(-1)
	batch := 1
	if ids.NumDims() == 2 {
		batch = ids.GetDim(0)
	}
	out := C.cuda_malloc(C.size_t(batch * row * col * embd.ElemTypeSize() / embd.ElemBlockSize()))
	C.cuda_embedding(out, ids.dptr, embd.dptr, C.int(batch), C.int(row), C.int(col))
	ret := NewArray(Shape{batch, row, col}, embd.dtype)
	ret.dptr = out
	return ret, nil
}

func (r *cudaRunner) Rmsnorm(a, x *Array, eps float32) (*Array, error) {
	if x.GetDim(-1) != a.GetDim(-1) {
		return nil, fmt.Errorf("weight shape does not match")
	}
	ne := x.Len()
	col := x.GetDim(-1)
	N := ne / col
	out := C.cuda_malloc(C.size_t(ne * x.ElemTypeSize() / x.ElemBlockSize()))
	C.cuda_rmsnorm(out, x.dptr, a.dptr, C.int(N), C.int(col), C.float(eps))
	ret := NewArray(x.Shape, x.dtype)
	ret.dptr = out
	return ret, nil
}

func (r *cudaRunner) Cat(a, b *Array) (*Array, error) {
	if a.dtype != b.dtype {
		return nil, fmt.Errorf("arrays must have the same dtype")
	}
	if a.NumDims() != 2 || b.NumDims() != 2 {
		return nil, fmt.Errorf("arrays must have 2 dimensions")
	}
	if a.GetDim(-1) != b.GetDim(-1) {
		return nil, fmt.Errorf("array shapes do not match")
	}
	col := a.GetDim(-1)
	arow := a.GetDim(0)
	brow := b.GetDim(0)
	out := C.cuda_malloc(C.size_t((arow + brow) * col * a.ElemTypeSize() / a.ElemBlockSize()))
	C.cuda_cat(out, a.dptr, b.dptr, C.int(arow), C.int(brow), C.int(col))
	shape := a.Shape
	shape[0] = arow + brow
	ret := NewArray(shape, a.dtype)
	ret.dptr = out
	return ret, nil
}

func slicesEqual(slice1, slice2 []int) bool {
	if len(slice1) != len(slice2) {
		return false
	}

	for i := range slice1 {
		if slice1[i] != slice2[i] {
			return false
		}
	}

	return true
}

func (r *cudaRunner) DivInPlace(a, b *Array) error {
	if a.dtype != b.dtype {
		return fmt.Errorf("arrays must have the same dtype")
	}
	if !slicesEqual(a.Shape, b.Shape) {
		return fmt.Errorf("array shapes do not match")
	}
	col := a.GetDim(-1)
	row := a.GetDim(0)
	C.cuda_div(a.dptr, a.dptr, b.dptr, C.int(row), C.int(col))
	return nil
}
