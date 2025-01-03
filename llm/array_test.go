package llm

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"testing"
	"unsafe"

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
	s = Shape{}
	assert.Equal(t, s.Len(), 1) // No dimensions means a scalar
}

func TestTensorElemSize(t *testing.T) {
	tests := []struct {
		shape    Shape
		data     any
		expected int
	}{
		{Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6}, 4},
		{Shape{2, 3}, []int8{1, 2, 3, 4, 5, 6}, 1},
	}

	for _, tt := range tests {
		a, err := MakeTensor(tt.shape, tt.data)
		assert.NoErr(t, err)
		assert.NotNil(t, a)
		assert.Equal(t, a.ElemTypeSize()/a.ElemBlockSize(), tt.expected)
	}
}

func TestTensorSize(t *testing.T) {
	tests := []struct {
		shape    Shape
		data     any
		expected int
	}{
		{Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6}, 24},
		{Shape{2, 3}, []int16{1, 2, 3, 4, 5, 6}, 12},
	}

	for _, tt := range tests {
		a, err := MakeTensor(tt.shape, tt.data)
		assert.NoErr(t, err)
		assert.Equal(t, a.Size(), tt.expected)
	}
}

func TestShapeNumDims(t *testing.T) {
	s := Shape{2, 3, 4}
	expected := 3
	assert.Equal(t, s.NumDims(), expected)
}

func TestShapeGetDim(t *testing.T) {
	s := Shape{2, 3, 4}
	assert.Equal(t, s.GetDim(0), 2)
	assert.Equal(t, s.GetDim(1), 3)
	assert.Equal(t, s.GetDim(2), 4)
	assert.Equal(t, s.GetDim(-1), 4)
	assert.Equal(t, s.GetDim(-2), 3)
	assert.Equal(t, s.GetDim(-3), 2)
	assert.Panic(t, func() { s.GetDim(3) })
	assert.Panic(t, func() { s.GetDim(-4) })
}

func TestShapeSetDim(t *testing.T) {
	s := Shape{2, 3, 4}
	s.SetDim(0, 5)
	s.SetDim(1, 6)
	s.SetDim(2, 7)
	assert.Equal(t, s.GetDim(0), 5)
	assert.Equal(t, s.GetDim(1), 6)
	assert.Equal(t, s.GetDim(2), 7)
	s.SetDim(-1, 8)
	assert.Equal(t, s.GetDim(2), 8)
	assert.Panic(t, func() { s.SetDim(5, 8) })
	assert.Panic(t, func() { s.SetDim(-5, 8) })
}

func TestShapeIsScalar(t *testing.T) {
	s := Shape{}
	assert.True(t, s.IsScalar())

	s = Shape{1}
	assert.False(t, s.IsScalar())
}

func TestTensorFormat(t *testing.T) {
	tests := []struct {
		shape    Shape
		data     any
		expected string
	}{
		{
			Shape{2, 2, 3},
			[]int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			"Shape: (2, 2, 3)\nType: Int16\nData:\n 1 2 3\n 4 5 6\n\n 7 8 9\n 10 11 12\n",
		},
		{
			Shape{2, 3},
			[]float32{1.1, 2.2, 3.3, 4.4, 5.5, 6.6},
			"Shape: (2, 3)\nType: F32\nData:\n 1.1 2.2 3.3\n 4.4 5.5 6.6\n",
		},
	}

	for _, tt := range tests {
		a, err := MakeTensor(tt.shape, tt.data)
		assert.NoErr(t, err)
		result := fmt.Sprintf("%v", a)
		assert.Equal(t, tt.expected, result)
	}
	// runtime.GC() // test if finalizer is called
}

func TestShapeFormat(t *testing.T) {
	s := Shape{2, 3, 4}
	expected := "(2, 3, 4)"
	result := fmt.Sprintf("%v", s)
	assert.Equal(t, result, expected)
}

func TestMakeTensor(t *testing.T) {
	tests := []struct {
		Shape Shape
		data  any
		valid bool
	}{
		{Shape{4, 2, 2}, randFloatSlice(16), true},
		{Shape{2, 3}, randFloatSlice(6), true},
		{Shape{2, 2}, []int{1, 2, 3}, false},
		{Shape{}, nil, false},
		{Shape{}, []int{1, 2, 3}, false},
	}

	for _, tt := range tests {
		_, err := MakeTensor(tt.Shape, tt.data)
		assert.Equal(t, err == nil, tt.valid)
	}
}

func TestTensorDeviceFree(t *testing.T) {
	var text string
	shape := Shape{2, 3}
	v := reflect.ValueOf([]float32{1, 2, 3, 4, 5, 6})
	a := &Tensor{shape, Storage{
		dtype: DTypeOf(v.Type().Elem()),
	}, &cudaRunner{}}
	a.ToDevice(unsafe.Pointer(v.Pointer()))
	assert.NotNil(t, a.dptr)
	assert.Equal(t, []float32{1, 2, 3, 4, 5, 6}, a.ToHost().([]float32))

	runtime.SetFinalizer(a, func(a *Tensor) {
		a.DeviceFree()
		assert.Nil(t, a.dptr)
		text = "DeviceFree called"
	})
	a = nil
	runtime.GC()
	runtime.GC() // call gc twice to ensure finalizer is called.
	assert.Equal(t, "DeviceFree called", text)
}

func TestTensorSoftmax(t *testing.T) {
	tests := []struct {
		shape    Shape
		input    []float32
		expected []float32
	}{
		{Shape{1, 3}, []float32{1.0, 1.0, 1.0}, []float32{0.33333333, 0.33333333, 0.33333333}},
		{Shape{2, 1, 3}, []float32{3.0, 1.0, 0.2, 1, 1000, 2}, []float32{0.8360188, 0.11314284, 0.05083836, 0.0, 1.0, 0.0}},
	}

	a, err := MakeTensor(Shape{3}, []float32{1.0, 2.0, 3.0})
	assert.NoErr(t, err)
	res, err := a.Softmax()
	assert.NoErr(t, err)
	assert.SliceNear(t, []float32{0.0900305, 0.2447284, 0.6652409}, res.ToHost().([]float32), 1e-6)

	for _, tt := range tests {
		array, err := MakeTensor(tt.shape, tt.input)
		assert.NoErr(t, err)

		res, err := array.Softmax()
		assert.NoErr(t, err)
		assert.SliceNear(t, tt.expected, res.ToHost().([]float32), 1e-6)
	}
}

func TestTensorMatmul(t *testing.T) {
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
			aData: []float32{1, 2, 3, 4, 5, 6}, bData: []float32{7, 8, 9, 10},
			expected: nil, expectError: true,
		},
		{
			a: Shape{2, 3}, b: Shape{2, 2, 2}, bias: Shape{},
			aData: []float32{1, 2, 3, 4, 5, 6}, bData: []float32{7, 8, 9, 10, 11, 12, 13, 14}, biasData: nil,
			expected: nil, expectError: true,
		},
		{
			a: Shape{2, 3}, b: Shape{2, 3}, bias: Shape{3},
			aData: []float32{1, 2, 3, 4, 5, 6}, bData: []float32{7, 8, 9, 10, 11, 12}, biasData: []float32{1, 1, 1},
			expected: nil, expectError: true,
		},
	}

	for _, tt := range tests {
		a, err := MakeTensor(tt.a, tt.aData)
		assert.NoErr(t, err)
		b, err := MakeTensor(tt.b, tt.bData)
		assert.NoErr(t, err)
		var bias *Tensor
		if tt.biasData != nil {
			bias, err = MakeTensor(tt.bias, tt.biasData)
			assert.NoErr(t, err)
		}

		result, err := a.Matmul(b, bias)
		assert.Equal(t, err != nil, tt.expectError)
		if err != nil {
			continue
		}
		assert.SliceNear(t, tt.expected, result.ToHost().([]float32), 1e-6)
	}
	row := 4
	col := 32
	e := NewTensor(Shape{row, col}, GGML_TYPE_Q8_0)
	data := []uint8{
		8, 52, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127,
		0, 25, 0, 252, 8, 244, 16, 236, 25, 227, 33, 219, 41, 211, 49, 203, 57, 195, 66, 186, 74, 178, 82, 170, 90, 162, 98, 154, 107, 145, 115, 137, 123, 129,
		0, 12, 0, 4, 8, 12, 16, 20, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 107, 111, 115, 119, 123, 127,
		154, 1, 0, 4, 8, 12, 16, 20, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 107, 111, 115, 119, 123, 127,
	}
	e.ToDevice(unsafe.Pointer(&data[0]))
	ones := MakeTensorIdentity(32)
	r, err := ones.Matmul(e, nil)
	assert.NoErr(t, err)
	expected := []float32{
		1.0078125, 0, 0, 0,
		2.015625, -0.009765625, 0.0009765625, 9.775162e-05,
		3.0234375, 0.01953125, 0.001953125, 0.00019550323,
		4.03125, -0.029296875, 0.0029296875, 0.00029325485,
		5.0390625, 0.0390625, 0.00390625, 0.00039100647,
		6.046875, -0.048828125, 0.0048828125, 0.0004887581,
		7.0546875, 0.061035156, 0.0061035156, 0.0006109476,
		8.0625, -0.07080078, 0.007080078, 0.0007086992,
		9.0703125, 0.080566406, 0.008056641, 0.00080645084,
		10.078125, -0.09033203, 0.009033203, 0.00090420246,
		11.0859375, 0.100097656, 0.010009766, 0.0010019541,
		12.09375, -0.10986328, 0.010986328, 0.0010997057,
		13.1015625, 0.119628906, 0.011962891, 0.0011974573,
		14.109375, -0.12939453, 0.012939453, 0.0012952089,
		15.1171875, 0.13916016, 0.013916016, 0.0013929605,
		16.125, -0.14892578, 0.014892578, 0.0014907122,
		16.88086, 0.16113281, 0.016113281, 0.0016129017,
		17.888672, -0.17089844, 0.017089844, 0.0017106533,
		18.896484, 0.18066406, 0.018066406, 0.0018084049,
		19.904297, -0.19042969, 0.019042969, 0.0019061565,
		20.91211, 0.20019531, 0.020019531, 0.0020039082,
		21.919922, -0.20996094, 0.020996094, 0.0021016598,
		22.927734, 0.21972656, 0.021972656, 0.0021994114,
		23.935547, -0.22949219, 0.022949219, 0.002297163,
		24.94336, 0.23925781, 0.023925781, 0.0023949146,
		25.951172, -0.24902344, 0.024902344, 0.0024926662,
		26.958984, 0.26123047, 0.026123047, 0.0026148558,
		27.966797, -0.2709961, 0.02709961, 0.0027126074,
		28.97461, 0.28076172, 0.028076172, 0.002810359,
		29.982422, -0.29052734, 0.029052734, 0.0029081106,
		30.990234, 0.30029297, 0.030029297, 0.0030058622,
		31.998047, -0.3100586, 0.03100586, 0.0031036139,
	}
	assert.SliceNear(t, expected, r.ToHost().([]float32), 1e-6)
}

func TestTensorMatmulSoftmax(t *testing.T) {
	a, err := MakeTensor(Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	assert.NoErr(t, err)
	b, err := MakeTensor(Shape{3, 3}, []float32{1, 4, 7, 2, 5, 8, 3, 6, 9})
	assert.NoErr(t, err)
	c, err := a.Matmul(b, nil)
	assert.NoErr(t, err)
	d, err := c.Softmax()
	assert.NoErr(t, err)
	expected := []float32{6.1289825e-06, 0.0024726081, 0.9975212, 9.3576195e-14, 3.059022e-07, 0.99999964}
	assert.SliceNear(t, expected, d.ToHost().([]float32), 1e-6)
}

func TestTensorEmbedding(t *testing.T) {
	tests := []struct {
		embdShape, idsShape Shape
		embdData, idsData   any
		expected            []float32
		expectError         bool
	}{
		{
			embdShape: Shape{3, 4}, idsShape: Shape{2},
			embdData: []float32{
				0.1, 0.2, 0.3, 0.4,
				0.5, 0.6, 0.7, 0.8,
				0.9, 1.0, 1.1, 1.2,
			},
			idsData:     []int32{1, 2},
			expected:    []float32{0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2},
			expectError: false,
		},
		{
			embdShape: Shape{4, 4}, idsShape: Shape{2, 2},
			embdData: []float32{
				0.1, 0.2, 0.3, 0.4,
				0.5, 0.6, 0.7, 0.8,
				0.9, 1.0, 1.1, 1.2,
				1.3, 1.4, 1.5, 1.6,
			},
			idsData:     []int32{1, 3, 0, 2},
			expected:    []float32{0.5, 0.6, 0.7, 0.8, 1.3, 1.4, 1.5, 1.6, 0.1, 0.2, 0.3, 0.4, 0.9, 1.0, 1.1, 1.2},
			expectError: false,
		},
		{
			embdShape: Shape{4}, idsShape: Shape{3},
			embdData:    []float32{0.1, 0.2, 0.3, 0.4},
			idsData:     []int32{1, 3, 4},
			expected:    nil,
			expectError: true,
		},
		{
			embdShape: Shape{2, 4}, idsShape: Shape{2, 3},
			embdData:    []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
			idsData:     []int64{1, 3, 0, 2, 1, 3},
			expected:    nil,
			expectError: true,
		},
		{
			embdShape: Shape{2, 2, 2}, idsShape: Shape{2, 3},
			embdData:    []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
			idsData:     []int32{1, 3, 0, 2, 1, 3},
			expected:    nil,
			expectError: true,
		},
		{
			embdShape: Shape{2, 4}, idsShape: Shape{2, 2, 2},
			embdData:    []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
			idsData:     []int32{1, 3, 0, 2, 1, 3, 0, 2},
			expected:    nil,
			expectError: true,
		},
	}

	for _, tt := range tests {
		embd, err := MakeTensor(tt.embdShape, tt.embdData)
		assert.NoErr(t, err)
		ids, err := MakeTensor(tt.idsShape, tt.idsData)
		assert.NoErr(t, err)

		result, err := embd.Embedding(ids)
		assert.Equal(t, err != nil, tt.expectError)
		if err != nil {
			continue
		}
		assert.SliceNear(t, tt.expected, result.ToHost().([]float32), 1e-6)
	}

	row := 4
	col := 32
	e := NewTensor(Shape{row, col}, GGML_TYPE_Q8_0)
	data := []uint8{
		8, 52, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127,
		0, 25, 0, 252, 8, 244, 16, 236, 25, 227, 33, 219, 41, 211, 49, 203, 57, 195, 66, 186, 74, 178, 82, 170, 90, 162, 98, 154, 107, 145, 115, 137, 123, 129,
		8, 52, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127,
		0, 25, 0, 252, 8, 244, 16, 236, 25, 227, 33, 219, 41, 211, 49, 203, 57, 195, 66, 186, 74, 178, 82, 170, 90, 162, 98, 154, 107, 145, 115, 137, 123, 129,
	}
	e.ToDevice(unsafe.Pointer(&data[0]))
	ids, err := MakeTensor(Shape{2}, []int32{2, 1})
	assert.NoErr(t, err)
	r, err := e.Embedding(ids)
	assert.NoErr(t, err)
	expected := []float32{
		1.0078125, 2.015625, 3.0234375, 4.03125, 5.0390625, 6.046875, 7.0546875, 8.0625, 9.0703125, 10.078125, 11.0859375, 12.09375, 13.1015625, 14.109375, 15.1171875, 16.125, 16.88086, 17.888672, 18.896484, 19.904297, 20.91211, 21.919922, 22.927734, 23.935547, 24.94336, 25.951172, 26.958984, 27.966797, 28.97461, 29.982422, 30.990234, 31.998047,
		0, -0.009765625, 0.01953125, -0.029296875, 0.0390625, -0.048828125, 0.061035156, -0.07080078, 0.080566406, -0.09033203, 0.100097656, -0.10986328, 0.119628906, -0.12939453, 0.13916016, -0.14892578, 0.16113281, -0.17089844, 0.18066406, -0.19042969, 0.20019531, -0.20996094, 0.21972656, -0.22949219, 0.23925781, -0.24902344, 0.26123047, -0.2709961, 0.28076172, -0.29052734, 0.30029297, -0.3100586,
	}
	assert.SliceNear(t, expected, r.ToHost().([]float32), 1e-6)
}

func TestMakeTensorFrom(t *testing.T) {
	tests := []struct {
		name        string
		shape       Shape
		data        []float32
		dtype       DType
		expectError bool
	}{
		{
			name:        "Valid float32 array",
			shape:       Shape{2, 3},
			data:        []float32{1, 2, 3, 4, 5, 6},
			dtype:       GGML_TYPE_F32,
			expectError: false,
		},
		{
			name:        "Empty shape",
			shape:       Shape{},
			data:        []float32{1},
			dtype:       GGML_TYPE_F32,
			expectError: true,
		},
		{
			name:        "Nil shape",
			shape:       nil,
			data:        []float32{1},
			dtype:       GGML_TYPE_F32,
			expectError: true,
		},
		{
			name:        "Nil data",
			shape:       Shape{2, 3},
			data:        nil,
			dtype:       GGML_TYPE_F32,
			expectError: true,
		},
		{
			name:        "Zeroed shape",
			shape:       Shape{0, 3},
			data:        []float32{1, 2, 3},
			dtype:       GGML_TYPE_F32,
			expectError: true,
		},
		{
			name:        "Invalid dtype",
			shape:       Shape{2, 3},
			data:        []float32{1, 2, 3, 4, 5, 6},
			dtype:       DType(9999), // Invalid dtype
			expectError: true,
		},
		{
			name:        "Misaligned shape for Q8_0",
			shape:       Shape{2, 12},
			data:        []float32{1, 2, 3, 4, 5, 6},
			dtype:       GGML_TYPE_Q8_0,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ptr unsafe.Pointer
			if tt.data != nil {
				ptr = unsafe.Pointer(&tt.data[0])
			}
			arr, err := MakeTensorFrom(tt.shape, ptr, tt.dtype)

			if tt.expectError {
				assert.Error(t, err)
				return
			}

			assert.NoErr(t, err)
			assert.NotNil(t, arr)
			assert.Equal(t, arr.Shape, tt.shape)
			assert.Equal(t, arr.dtype, tt.dtype)
		})
	}
}

func TestTensorRmsnorm(t *testing.T) {
	tests := []struct {
		x        []float32
		w        []float32
		xShape   Shape
		wShape   Shape
		eps      float32
		wantErr  bool
		expected []float32
	}{
		{
			x:        []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
			w:        []float32{0.5, 0.5, 0.5},
			xShape:   Shape{2, 3},
			wShape:   Shape{3},
			eps:      1e-6,
			wantErr:  false,
			expected: []float32{0.231455, 0.462910, 0.694365, 0.394771, 0.493463, 0.592156},
		},
		{
			x:       []float32{1.0, 2.0, 3.0, 4.0},
			w:       []float32{0.5, 0.5, 0.5},
			xShape:  Shape{2, 2},
			wShape:  Shape{3},
			eps:     1e-6,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		x, err := MakeTensor(tt.xShape, tt.x)
		assert.NoErr(t, err)

		w, err := MakeTensor(tt.wShape, tt.w)
		assert.NoErr(t, err)

		result, err := w.Rmsnorm(x, tt.eps)
		if tt.wantErr {
			assert.Error(t, err)
			return
		}
		assert.NoErr(t, err)
		assert.Equal(t, x.Shape, result.Shape)
		assert.SliceNear(t, tt.expected, result.ToHost().([]float32), 1e-6)
	}
}
func TestTensorCat(t *testing.T) {
	tests := []struct {
		aShape, bShape Shape
		aData, bData   []float32
		expectedShape  Shape
		expectedData   []float32
		expectError    bool
	}{
		{
			aShape: Shape{2, 3}, bShape: Shape{2, 3},
			aData: []float32{1, 2, 3, 4, 5, 6}, bData: []float32{7, 8, 9, 10, 11, 12},
			expectedShape: Shape{4, 3},
			expectedData:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			expectError:   false,
		},
		{
			aShape: Shape{2, 3}, bShape: Shape{1, 3},
			aData: []float32{1, 2, 3, 4, 5, 6}, bData: []float32{7, 8, 9},
			expectedShape: Shape{3, 3},
			expectedData:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			expectError:   false,
		},
		{
			aShape: Shape{2, 3}, bShape: Shape{2, 2},
			aData: []float32{1, 2, 3, 4, 5, 6}, bData: []float32{7, 8, 9, 10},
			expectedShape: nil,
			expectedData:  nil,
			expectError:   true,
		},
	}

	for _, tt := range tests {
		a, err := MakeTensor(tt.aShape, tt.aData)
		assert.NoErr(t, err)
		b, err := MakeTensor(tt.bShape, tt.bData)
		assert.NoErr(t, err)

		result, err := a.Cat(b)
		assert.Equal(t, tt.expectError, err != nil)
		if err != nil {
			continue
		}

		assert.Equal(t, tt.expectedShape, result.Shape)
		assert.Equal(t, tt.expectedData, result.ToHost().([]float32))
	}
	// Test different data types
	a, err := MakeTensor(Shape{3, 2}, []int32{1, 2, 3, 4, 5, 6})
	assert.NoErr(t, err)
	b, err := MakeTensor(Shape{2, 3}, []float32{7, 8, 9, 10, 11, 12})
	assert.NoErr(t, err)
	_, err = a.Cat(b)
	assert.Error(t, err)
}

func TestTensorDivInPlace(t *testing.T) {
	tests := []struct {
		aShape, bShape Shape
		aData, bData   []float32
		expectedData   []float32
		expectError    bool
	}{
		{
			aShape: Shape{2, 3}, bShape: Shape{2, 3},
			aData: []float32{2, 4, 6, 8, 10, 12}, bData: []float32{1, 2, 3, 4, 5, 6},
			expectedData: []float32{2, 2, 2, 2, 2, 2},
			expectError:  false,
		},
		{
			aShape: Shape{2, 3}, bShape: Shape{2, 3},
			aData:        []float32{2, 4, 6, 8, 10, -12},
			bData:        []float32{1, 0, 3, 4, float32(math.Inf(1)), 0},
			expectedData: []float32{2, float32(math.Inf(1)), 2, 2, 0, float32(math.Inf(-1))},
			expectError:  false,
		},
		{
			aShape: Shape{2, 3}, bShape: Shape{2, 2},
			aData: []float32{2, 4, 6, 8, 10, 12}, bData: []float32{1, 2, 3, 4},
			expectedData: nil,
			expectError:  true,
		},
	}

	for _, tt := range tests {
		a, err := MakeTensor(tt.aShape, tt.aData)
		assert.NoErr(t, err)
		b, err := MakeTensor(tt.bShape, tt.bData)
		assert.NoErr(t, err)

		err = a.DivInPlace(b)
		assert.Equal(t, tt.expectError, err != nil)
		if err != nil {
			continue
		}

		assert.Equal(t, tt.expectedData, a.ToHost().([]float32))
	}
}

func TestTensorRopeInPlace(t *testing.T) {
	tests := []struct {
		aShape, posShape Shape
		aData            []float32
		bData            []float32
		expectedData     []float32
		expectError      bool
	}{
		{
			aShape:   Shape{2, 2},
			posShape: Shape{1},
			aData:    []float32{1.0, 2.0, 3.0, 4.0},
			bData:    []float32{1.0},
			expectedData: []float32{
				1.0, 2.0,
				-1.7449772, 4.685622,
			},
			expectError: false,
		},
		{
			aShape:   Shape{2, 2, 4},
			posShape: Shape{2},
			aData: []float32{
				1.0, 2.0, 3.0, 4.0,
				5.0, 6.0, 7.0, 8.0,
				9.0, 10.0, 11.0, 12.0,
				13.0, 14.0, 15.0, 16.0,
			},
			bData: []float32{1.0, 0.01},
			expectedData: []float32{
				1.000000, 2.000000, 3.000000, 4.000000,
				-2.34731436, 7.44916821, 6.91965151, 8.06959915,
				9.000000, 10.000000, 11.000000, 12.000000,
				-4.75666428, 18.50335503, 14.83925247, 16.14919662,
			},
			expectError: false,
		},
		{
			aShape:       Shape{2, 2},
			posShape:     Shape{2, 1},
			aData:        []float32{1.0, 2.0, 3.0, 4.0},
			bData:        []float32{1.0, 1.0},
			expectedData: nil,
			expectError:  true,
		},
	}

	for _, tt := range tests {
		a, err := MakeTensor(tt.aShape, tt.aData)
		assert.NoErr(t, err)
		freqs, err := MakeTensor(tt.posShape, tt.bData)
		assert.NoErr(t, err)

		err = a.RopeInPlace(freqs)
		assert.Equal(t, tt.expectError, err != nil)
		if err != nil {
			continue
		}

		result := a.ToHost().([]float32)
		if tt.expectedData != nil {
			assert.SliceNear(t, tt.expectedData, result, 1e-6)
		}
	}
	a, err := MakeTensor(Shape{2, 2}, []float32{1, 2, 3, 4})
	assert.NoErr(t, err)
	freqs, err := MakeTensor(Shape{1}, []int{1})
	assert.NoErr(t, err)
	err = a.RopeInPlace(freqs)
	assert.Error(t, err)
}

func TestTensorDequantize(t *testing.T) {
	tests := []struct {
		shape       Shape
		data        any
		dtype       DType
		expectError bool
		expected    []float32
	}{
		{
			shape: Shape{2, 32},
			data: []uint8{
				8, 52, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127,
				0, 25, 0, 252, 8, 244, 16, 236, 25, 227, 33, 219, 41, 211, 49, 203, 57, 195, 66, 186, 74, 178, 82, 170, 90, 162, 98, 154, 107, 145, 115, 137, 123, 129,
			},
			dtype:       GGML_TYPE_Q8_0,
			expectError: false,
			expected: []float32{
				1.0078125, 2.015625, 3.0234375, 4.03125, 5.0390625, 6.046875, 7.0546875, 8.0625, 9.0703125, 10.078125, 11.0859375, 12.09375, 13.1015625, 14.109375, 15.1171875, 16.125, 16.88086, 17.888672, 18.896484, 19.904297, 20.91211, 21.919922, 22.927734, 23.935547, 24.94336, 25.951172, 26.958984, 27.966797, 28.97461, 29.982422, 30.990234, 31.998047,
				0, -0.009765625, 0.01953125, -0.029296875, 0.0390625, -0.048828125, 0.061035156, -0.07080078, 0.080566406, -0.09033203, 0.100097656, -0.10986328, 0.119628906, -0.12939453, 0.13916016, -0.14892578, 0.16113281, -0.17089844, 0.18066406, -0.19042969, 0.20019531, -0.20996094, 0.21972656, -0.22949219, 0.23925781, -0.24902344, 0.26123047, -0.2709961, 0.28076172, -0.29052734, 0.30029297, -0.3100586,
			},
		},
		{
			shape: Shape{2, 32},
			data: []uint8{
				8, 52, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127,
				0, 25, 0, 252, 8, 244, 16, 236, 25, 227, 33, 219, 41, 211, 49, 203, 57, 195, 66, 186, 74, 178, 82, 170, 90, 162, 98, 154, 107, 145, 115, 137, 123, 129,
			},
			dtype:       GGML_TYPE_F32,
			expectError: true,
			expected:    nil,
		},
	}

	for _, tt := range tests {
		a := NewTensor(tt.shape, tt.dtype)
		v := reflect.ValueOf(tt.data)
		a.ToDevice(unsafe.Pointer(v.Pointer()))
		d, err := a.Dequantize()
		if tt.expectError {
			assert.Error(t, err)
			return
		}
		assert.NoErr(t, err)
		assert.SliceNear(t, tt.expected, d.ToHost().([]float32), 1e-6)
	}
}

func BenchmarkTensor(b *testing.B) {
	row := 128256
	col := 8192
	t := NewTensor(Shape{row, col}, GGML_TYPE_Q8_0)
	info := DTypeInfo[GGML_TYPE_Q8_0]
	data := make([]uint8, row*col/info.blockSize*info.typeSize)
	r := rand.New(rand.NewSource(0))
	r.Read(data)
	t.ToDevice(unsafe.Pointer(&data[0]))
	b.Run("Dequantize", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			a, err := t.Dequantize()
			assert.NoErr(b, err)
			a.DeviceFree()
		}
	})
}

func TestMakeTensorOnes(t *testing.T) {
	s := Shape{2, 3}
	a := MakeTensorOnes(s)
	expected := []float32{1, 1, 1, 1, 1, 1}
	assert.Equal(t, a.Shape, s)
	assert.Equal(t, expected, a.ToHost())
}

func TestMakeTensorIdentity(t *testing.T) {
	d := 3
	a := MakeTensorIdentity(d)
	expected := []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	}
	assert.Equal(t, a.Shape, Shape{d, d})
	assert.Equal(t, expected, a.ToHost())
}

func TestEndToEndInference(t *testing.T) {
	batch := 2
	row := 4
	col := 8
	NH := 2
	kvNH := 1
	HS := 4
	ffl := 16
	vocab := 16
	eps := float32(1e-5)

	embedding, err := MakeTensor(Shape{vocab, col}, []float32{
		0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4,
		0.2, 0.3, 0.4, 0.5, -0.2, -0.3, -0.4, -0.5,
		0.3, 0.4, 0.5, 0.6, -0.3, -0.4, -0.5, -0.6,
		0.4, 0.5, 0.6, 0.7, -0.4, -0.5, -0.6, -0.7,
		0.5, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8,
		0.6, 0.7, 0.8, 0.9, -0.6, -0.7, -0.8, -0.9,
		0.7, 0.8, 0.9, 1.0, -0.7, -0.8, -0.9, -1.0,
		0.8, 0.9, 1.0, 0.1, -0.8, -0.9, -1.0, -0.1,
		0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4,
		0.2, 0.3, 0.4, 0.5, -0.2, -0.3, -0.4, -0.5,
		0.3, 0.4, 0.5, 0.6, -0.3, -0.4, -0.5, -0.6,
		0.4, 0.5, 0.6, 0.7, -0.4, -0.5, -0.6, -0.7,
		0.5, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8,
		0.6, 0.7, 0.8, 0.9, -0.6, -0.7, -0.8, -0.9,
		0.7, 0.8, 0.9, 1.0, -0.7, -0.8, -0.9, -1.0,
		0.8, 0.9, 1.0, 0.1, -0.8, -0.9, -1.0, -0.1,
	})
	assert.NoErr(t, err)

	ids, err := MakeTensor(Shape{batch, row}, []int32{
		1, 2, 3, 4,
		5, 6, 7, 0,
	})
	assert.NoErr(t, err)

	freqs, err := MakeTensor(Shape{HS / 2}, []float32{1.0, 0.01})
	assert.NoErr(t, err)

	attn_norm, err := MakeTensor(Shape{col}, []float32{
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
	})
	assert.NoErr(t, err)

	// QKV weights (NH + 2*kvNH) * HS, col)
	qkv_weight, err := MakeTensor(Shape{(NH + 2*kvNH) * HS, col}, []float32{
		// Q weights
		0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4,
		-0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.4, 0.4,
		0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5,
		-0.2, 0.2, -0.3, 0.3, -0.4, 0.4, -0.5, 0.5,
		0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4,
		-0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.4, 0.4,
		0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5,
		-0.2, 0.2, -0.3, 0.3, -0.4, 0.4, -0.5, 0.5,
		// K weights
		0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6,
		-0.3, 0.3, -0.4, 0.4, -0.5, 0.5, -0.6, 0.6,
		0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6,
		-0.3, 0.3, -0.4, 0.4, -0.5, 0.5, -0.6, 0.6,
		// V weights
		0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7,
		-0.4, 0.4, -0.5, 0.5, -0.6, 0.6, -0.7, 0.7,
		0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7,
		-0.4, 0.4, -0.5, 0.5, -0.6, 0.6, -0.7, 0.7,
	})
	assert.NoErr(t, err)

	// Attention output projection (col, col)
	attn_out, err := MakeTensor(Shape{col, col}, []float32{
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
		0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
		0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
		0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1,
		0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2,
		0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3,
		0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4,
		0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5,
	})
	assert.NoErr(t, err)

	// Feed-forward weights
	ff_norm, err := MakeTensor(Shape{col}, []float32{
		0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
	})
	assert.NoErr(t, err)

	// Feed-forward weights (2*ffl, col)
	ff_weight, err := MakeTensor(Shape{2 * ffl, col}, []float32{
		// First half for FC1
		0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4,
		-0.2, 0.2, -0.3, 0.3, -0.4, 0.4, -0.5, 0.5,
		0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6,
		-0.4, 0.4, -0.5, 0.5, -0.6, 0.6, -0.7, 0.7,
		0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4,
		-0.2, 0.2, -0.3, 0.3, -0.4, 0.4, -0.5, 0.5,
		0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6,
		0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5,
		-0.3, 0.3, -0.4, 0.4, -0.5, 0.5, -0.6, 0.6,
		0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7,
		-0.5, 0.5, -0.6, 0.6, -0.7, 0.7, -0.8, 0.8,
		0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5,
		-0.3, 0.3, -0.4, 0.4, -0.5, 0.5, -0.6, 0.6,
		0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7,
		-0.5, 0.5, -0.6, 0.6, -0.7, 0.7, -0.8, 0.8,
		-0.4, 0.4, -0.5, 0.5, -0.6, 0.6, -0.7, 0.7,
		// Second half for FC2 (gate)
		0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5,
		-0.3, 0.3, -0.4, 0.4, -0.5, 0.5, -0.6, 0.6,
		0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7,
		-0.5, 0.5, -0.6, 0.6, -0.7, 0.7, -0.8, 0.8,
		0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5,
		-0.3, 0.3, -0.4, 0.4, -0.5, 0.5, -0.6, 0.6,
		0.4, -0.4, 0.5, -0.5, 0.6, -0.6, 0.7, -0.7,
		-0.5, 0.5, -0.6, 0.6, -0.7, 0.7, -0.8, 0.8,
		-0.4, 0.4, -0.5, 0.5, -0.6, 0.6, -0.7, 0.7,
		0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4,
		-0.2, 0.2, -0.3, 0.3, -0.4, 0.4, -0.5, 0.5,
		0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6,
		-0.4, 0.4, -0.5, 0.5, -0.6, 0.6, -0.7, 0.7,
		0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4,
		-0.2, 0.2, -0.3, 0.3, -0.4, 0.4, -0.5, 0.5,
		0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6,
	})
	assert.NoErr(t, err)

	// Feed-forward output projection (col, ffl)
	ff_out, err := MakeTensor(Shape{col, ffl}, []float32{
		0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4, 0.5, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8,
		0.2, 0.3, 0.4, 0.5, -0.2, -0.3, -0.4, -0.5, 0.6, 0.7, 0.8, 0.9, -0.6, -0.7, -0.8, -0.9,
		0.3, 0.4, 0.5, 0.6, -0.3, -0.4, -0.5, -0.6, 0.7, 0.8, 0.9, 1.0, -0.7, -0.8, -0.9, -1.0,
		0.4, 0.5, 0.6, 0.7, -0.4, -0.5, -0.6, -0.7, 0.8, 0.9, 1.0, 0.1, -0.8, -0.9, -1.0, -0.1,
		0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4, 0.5, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8,
		0.2, 0.3, 0.4, 0.5, -0.2, -0.3, -0.4, -0.5, 0.6, 0.7, 0.8, 0.9, -0.6, -0.7, -0.8, -0.9,
		0.3, 0.4, 0.5, 0.6, -0.3, -0.4, -0.5, -0.6, 0.7, 0.8, 0.9, 1.0, -0.7, -0.8, -0.9, -1.0,
		0.4, 0.5, 0.6, 0.7, -0.4, -0.5, -0.6, -0.7, 0.8, 0.9, 1.0, 0.1, -0.8, -0.9, -1.0, -0.1,
	})
	assert.NoErr(t, err)

	// Output classifier weights (vocab, col)
	classifier, err := MakeTensor(Shape{vocab, col}, []float32{
		0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4,
		0.2, 0.3, 0.4, 0.5, -0.2, -0.3, -0.4, -0.5,
		0.3, 0.4, 0.5, 0.6, -0.3, -0.4, -0.5, -0.6,
		0.4, 0.5, 0.6, 0.7, -0.4, -0.5, -0.6, -0.7,
		0.5, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8,
		0.6, 0.7, 0.8, 0.9, -0.6, -0.7, -0.8, -0.9,
		0.7, 0.8, 0.9, 1.0, -0.7, -0.8, -0.9, -1.0,
		0.8, 0.9, 1.0, 0.1, -0.8, -0.9, -1.0, -0.1,
		0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4,
		0.2, 0.3, 0.4, 0.5, -0.2, -0.3, -0.4, -0.5,
		0.3, 0.4, 0.5, 0.6, -0.3, -0.4, -0.5, -0.6,
		0.4, 0.5, 0.6, 0.7, -0.4, -0.5, -0.6, -0.7,
		0.5, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8,
		0.6, 0.7, 0.8, 0.9, -0.6, -0.7, -0.8, -0.9,
		0.7, 0.8, 0.9, 1.0, -0.7, -0.8, -0.9, -1.0,
		0.8, 0.9, 1.0, 0.1, -0.8, -0.9, -1.0, -0.1,
	})
	assert.NoErr(t, err)

	embeds, err := embedding.Embedding(ids)
	assert.NoErr(t, err)

	err = embeds.GroupQueryAttention(freqs, attn_norm, qkv_weight, attn_out, NH, kvNH, eps)
	assert.NoErr(t, err)

	ff_expected := []float32{
		0.32386804, 0.4635169, 0.60316575, 0.8296329, -0.076131955, 0.07035506, -0.19683424, -0.17036712,
		0.31848186, 0.4362287, 0.5539755, 0.8029859, -0.28151816, -0.17411, -0.44602448, -0.39701414,
		0.37628618, 0.48498565, 0.59368515, 0.8486642, -0.42371383, -0.33883628, -0.6063149, -0.55133575,
		0.45844948, 0.5630026, 0.6675557, 0.92151624, -0.5415505, -0.4716287, -0.7324443, -0.6784838,
		0.55500996, 0.65567404, 0.7563381, 0.9761183, -0.6449901, -0.6230009, -0.8436619, -0.8238816,
		0.65499586, 0.75531083, 0.8556257, 1.0715998, -0.7450041, -0.7280453, -0.94437426, -0.92840016,
		0.89583766, 1.0096093, 1.1233813, 0.13527647, -0.70416236, -0.8395548, -0.87661886, -0.06472353,
		0.08063001, 0.18098922, 0.2813484, 0.4337246, -0.11937, -0.16472037, -0.31865162, -0.3662754,
	}
	err = embeds.FeedForward(ff_norm, ff_weight, ff_out, ffl, eps)
	assert.NoErr(t, err)
	assert.Equal(t, ff_expected, embeds.ToHost())

	predictions := embeds.Predict(ff_norm, classifier, vocab, eps)

	expected := []int32{6, 6}
	assert.Equal(t, expected, predictions)
}
