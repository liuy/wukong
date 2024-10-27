package Array

/*
#cgo LDFLAGS: ./build/libwukong.a -L/usr/local/cuda/lib64 -lcudnn -lcublasLt -lcudart

void cuda_init(void);
void cuda_fini(void);
void cuda_to_host(void* dst, void* src, size_t size);
void cuda_to_device(void* dst, void* src, size_t size);
void* cuda_malloc(size_t size);
void cuda_free(void* ptr);
void cuda_matmul(void *out, const void *inp, const void *weight, const void *bias,
            int row, int column, int oc);
void cuda_softmax(void* output, void* intput, int row, int col);
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
	dptr  unsafe.Pointer
	dtype reflect.Type
}

type Runner interface {
	Softmax(a *Array) (*Array, error)
	Matmul(a, b, bias *Array) (*Array, error)
	ToDevice(a *Array, src unsafe.Pointer)
	ToHost(a *Array) reflect.Value
	DeviceFree(a *Array)
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

func NewArray(s Shape, t reflect.Type) *Array {
	a := &Array{
		s,
		Storage{
			dtype: t,
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
	if v.Len() != s.Size() {
		_, file, line, ok := runtime.Caller(1)
		if !ok {
			return nil, fmt.Errorf("data length does not match Shape")
		}
		return nil, fmt.Errorf("data length does not match Shape (file: %s, line: %d)", file, line)
	}
	t := v.Type()
	ret = NewArray(s, t)
	ret.ToDevice(unsafe.Pointer(v.Pointer()))
	return ret, nil
}

func (t *Array) Format(st fmt.State, r rune) {
	s := fmt.Sprintf("Shape: %v\nType: %v", t.Shape, t.dtype)

	data := "\nData:\n"
	dims := t.Shape
	stride := dims[len(dims)-1]

	d := t.ToHost()
	for i := 0; i < t.Size(); i++ {
		if i > 0 && i%stride == 0 {
			data += "\n"
			if i%(stride*dims[len(dims)-2]) == 0 {
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

// Run array operations on the CUDA device
type cudaRunner struct{}

func (r *cudaRunner) ToDevice(a *Array, src unsafe.Pointer) {
	if a.dptr == nil {
		a.dptr = C.cuda_malloc(C.size_t(a.Size() * int(a.dtype.Size())))
	}
	C.cuda_to_device(a.dptr, src, C.size_t(a.Size()*int(a.dtype.Size())))
}

func (r *cudaRunner) ToHost(a *Array) reflect.Value {
	dst := reflect.MakeSlice(a.dtype, a.Size(), a.Size())
	C.cuda_to_host(unsafe.Pointer(dst.Pointer()), a.dptr, C.size_t(a.Size()*int(a.dtype.Size())))
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
	if a.Dims() < 2 {
		return nil, fmt.Errorf("Array must have at least 2 dimensions")
	}
	col := a.Shape[len(a.Shape)-1]
	row := a.Size() / col
	out := C.cuda_malloc(C.size_t(row * col * int(a.dtype.Size())))
	C.cuda_softmax(out, a.dptr, C.int(row), C.int(col))
	ret := NewArray(a.Shape, a.dtype)
	ret.dptr = out
	return ret, nil
}

func (r *cudaRunner) Matmul(a, b, bias *Array) (*Array, error) {
	// For now we only support b as a 2D array
	if a.Dims() < 2 || b.Dims() != 2 {
		return nil, fmt.Errorf("arrays must have at least 2 dimensions")
	}
	if a.Shape[len(a.Shape)-1] != b.Shape[len(b.Shape)-2] {
		return nil, fmt.Errorf("array shapes do not match")
	}
	var biasPtr unsafe.Pointer = nil
	if bias != nil {
		biasPtr = bias.dptr
		if bias.Size() != b.Shape[len(b.Shape)-1] {
			return nil, fmt.Errorf("bias shape does not match")
		}
	}
	column := a.Shape[len(a.Shape)-1]
	row := a.Size() / column
	oc := b.Shape[len(b.Shape)-1]
	out := C.cuda_malloc(C.size_t(row * oc * int(a.dtype.Size())))

	C.cuda_matmul(out, a.dptr, b.dptr, biasPtr, C.int(row), C.int(column), C.int(oc))

	shape := a.Shape
	shape[len(shape)-1] = oc
	ret := NewArray(shape, a.dtype)
	ret.dptr = out
	return ret, nil
}
