package llm

import (
	"fmt"
	"math/rand"
	"os"
	"reflect"
	"runtime"
	"testing"
	"unsafe"

	"github.com/liuy/wukong/assert"
)

func TestMain(m *testing.M) {
	CudaSetup(0)
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
		a, err := MakeTensorErr(tt.shape, tt.data)
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
		a, err := MakeTensorErr(tt.shape, tt.data)
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
		a, err := MakeTensorErr(tt.shape, tt.data)
		assert.NoErr(t, err)
		result := fmt.Sprintf("%v", a)
		assert.Equal(t, tt.expected, result)
	}
	// runtime.GC() // test if finalizer is called
	a, err := MakeTensorErr(Shape{3, 3}, []float32{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9})
	assert.NoErr(t, err)
	res := fmt.Sprintf("%1r", a)
	assert.Equal(t, "Shape: (3, 3)\nType: F32\nData:\nRow 1:\n 1.1 2.2 3.3\n", res)
	res = fmt.Sprintf("%2r", a)
	assert.Equal(t, "Shape: (3, 3)\nType: F32\nData:\nRow 2:\n 4.4 5.5 6.6\n", res)
	res = fmt.Sprintf("%3r", a)
	assert.Equal(t, "Shape: (3, 3)\nType: F32\nData:\nRow 3:\n 7.7 8.8 9.9\n", res)
	res = fmt.Sprintf("%1c", a)
	assert.Equal(t, "Shape: (3, 3)\nType: F32\nData:\nColumn 1:\n 1.1\n 4.4\n 7.7\n", res)
	res = fmt.Sprintf("%2c", a)
	assert.Equal(t, "Shape: (3, 3)\nType: F32\nData:\nColumn 2:\n 2.2\n 5.5\n 8.8\n", res)
	res = fmt.Sprintf("%3c", a)
	assert.Equal(t, "Shape: (3, 3)\nType: F32\nData:\nColumn 3:\n 3.3\n 6.6\n 9.9\n", res)
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
		_, err := MakeTensorErr(tt.Shape, tt.data)
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

	a, err := MakeTensorErr(Shape{3}, []float32{1.0, 2.0, 3.0})
	assert.NoErr(t, err)
	res, err := a.Softmax()
	assert.NoErr(t, err)
	assert.SliceNear(t, []float32{0.0900305, 0.2447284, 0.6652409}, res.ToHost().([]float32), 1e-6)

	for _, tt := range tests {
		array, err := MakeTensorErr(tt.shape, tt.input)
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
		a, err := MakeTensorErr(tt.a, tt.aData)
		assert.NoErr(t, err)
		b, err := MakeTensorErr(tt.b, tt.bData)
		assert.NoErr(t, err)
		var bias *Tensor
		if tt.biasData != nil {
			bias, err = MakeTensorErr(tt.bias, tt.biasData)
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
		7.0546875, 0.061035156, 0.0061035156, 0.0006108284,
		8.0625, -0.07080078, 0.007080078, 0.00070858,
		9.0703125, 0.080566406, 0.008056641, 0.00080633163,
		10.078125, -0.09033203, 0.009033203, 0.00090408325,
		11.0859375, 0.100097656, 0.010009766, 0.0010023117,
		12.09375, -0.10986328, 0.010986328, 0.0010995865,
		13.1015625, 0.119628906, 0.011962891, 0.0011978149,
		14.109375, -0.12939453, 0.012939453, 0.0012950897,
		15.1171875, 0.13916016, 0.013916016, 0.0013933182,
		16.125, -0.14892578, 0.014892578, 0.001490593,
		16.875, 0.16113281, 0.016113281, 0.0016126633,
		17.890625, -0.17089844, 0.017089844, 0.0017108917,
		18.890625, 0.18066406, 0.018066406, 0.0018081665,
		19.90625, -0.19042969, 0.019042969, 0.001906395,
		20.90625, 0.20019531, 0.020019531, 0.0020046234,
		21.921875, -0.20996094, 0.020996094, 0.0021018982,
		22.921875, 0.21972656, 0.021972656, 0.002199173,
		23.9375, -0.22949219, 0.022949219, 0.0022964478,
		24.9375, 0.23925781, 0.023925781, 0.0023956299,
		25.953125, -0.24902344, 0.024902344, 0.0024929047,
		26.953125, 0.26123047, 0.026123047, 0.002614975,
		27.96875, -0.2709961, 0.02709961, 0.0027122498,
		28.96875, 0.28076172, 0.028076172, 0.0028095245,
		29.984375, -0.29052734, 0.029052734, 0.0029087067,
		30.984375, 0.30029297, 0.030029297, 0.0030059814,
		32, -0.3100586, 0.03100586, 0.0031032562,
	}
	assert.SliceNear(t, expected, r.ToHost().([]float32), 1e-6)
}

func TestTensorMatmulSoftmax(t *testing.T) {
	a, err := MakeTensorErr(Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	assert.NoErr(t, err)
	b, err := MakeTensorErr(Shape{3, 3}, []float32{1, 4, 7, 2, 5, 8, 3, 6, 9})
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
		embd, err := MakeTensorErr(tt.embdShape, tt.embdData)
		assert.NoErr(t, err)
		ids, err := MakeTensorErr(tt.idsShape, tt.idsData)
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
	ids, err := MakeTensorErr(Shape{2}, []int32{2, 1})
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
		a, err := MakeTensorErr(tt.aShape, tt.aData)
		assert.NoErr(t, err)
		b, err := MakeTensorErr(tt.bShape, tt.bData)
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
	a, err := MakeTensorErr(Shape{3, 2}, []int32{1, 2, 3, 4, 5, 6})
	assert.NoErr(t, err)
	b, err := MakeTensorErr(Shape{2, 3}, []float32{7, 8, 9, 10, 11, 12})
	assert.NoErr(t, err)
	_, err = a.Cat(b)
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

	embedding, err := MakeTensorErr(Shape{vocab, col}, []float32{
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

	ids, err := MakeTensorErr(Shape{batch, row}, []int32{
		1, 2, 3, 4,
		5, 6, 7, 0,
	})
	assert.NoErr(t, err)

	freqs, err := MakeTensorErr(Shape{row, HS}, []float32{
		1.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
		5.40302277e-01, 8.41471016e-01, 9.99999881e-01, 5.24846022e-04,
		-4.16146815e-01, 9.09297466e-01, 9.99999464e-01, 1.04969181e-03,
		-9.89992499e-01, 1.41120002e-01, 9.99998748e-01, 1.57453737e-03,
	})
	assert.NoErr(t, err)

	attn_norm, err := MakeTensorErr(Shape{col}, []float32{
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
	})
	assert.NoErr(t, err)

	// QKV weights (NH + 2*kvNH) * HS, col)
	qkv_weight, err := MakeTensorErr(Shape{(NH + 2*kvNH) * HS, col}, []float32{
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
	attn_out, err := MakeTensorErr(Shape{col, col}, []float32{
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
	ff_norm, err := MakeTensorErr(Shape{col}, []float32{
		0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
	})
	assert.NoErr(t, err)

	// Feed-forward weights (2*ffl, col)
	ff_weight, err := MakeTensorErr(Shape{2 * ffl, col}, []float32{
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
	ff_out, err := MakeTensorErr(Shape{col, ffl}, []float32{
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
	classifier, err := MakeTensorErr(Shape{vocab, col}, []float32{
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
		0.3241666, 0.463741, 0.60364395, 0.8300767, -0.07588394, 0.0707975, -0.19633079, -0.16992329,
		0.31827793, 0.4358318, 0.5536866, 0.8027342, -0.2817684, -0.1744478, -0.44629017, -0.39726582,
		0.37625444, 0.4847908, 0.5936067, 0.8485066, -0.4237886, -0.33904022, -0.6063718, -0.55149335,
		0.4586156, 0.5630526, 0.667752, 0.9216033, -0.54142475, -0.47152197, -0.7322278, -0.6783967,
		0.5550589, 0.65563756, 0.7564092, 0.97632813, -0.64497083, -0.62276554, -0.84357595, -0.8236718,
		0.6549731, 0.7551919, 0.8555962, 1.0717295, -0.74505544, -0.72791153, -0.94438946, -0.9282706,
		0.89586926, 1.0097064, 1.1234655, 0.13532515, -0.7041187, -0.83946335, -0.8765404, -0.064674854,
		0.08061107, 0.1809209, 0.28131697, 0.4337328, -0.11940219, -0.16472067, -0.31867644, -0.3662672,
	}
	err = embeds.FeedForward(ff_norm, ff_weight, ff_out, ffl, eps)
	assert.NoErr(t, err)
	assert.SliceNear(t, ff_expected, embeds.ToHost().([]float32), 1e-6)

	predictions := embeds.Predict(ff_norm, classifier, vocab, eps)

	expected := []int32{6, 6}
	assert.Equal(t, expected, predictions)
}

func TestSaveAndLoadTensor(t *testing.T) {
	tests := []struct {
		name        string
		shape       Shape
		data        []float32
		savePath    string
		expectError bool
	}{
		{
			name:        "valid tensor",
			shape:       Shape{2, 3},
			data:        []float32{1, 2, 3, 4, 5, 6},
			savePath:    "/tmp/test_tensor.bin",
			expectError: false,
		},
		{
			name:        "invalid path",
			shape:       Shape{2, 3},
			data:        []float32{1, 2, 3, 4, 5, 6},
			savePath:    "/nonexistent/dir/tensor.bin",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create and save tensor
			tensor, err := MakeTensorErr(tt.shape, tt.data)
			assert.NoErr(t, err)

			err = tensor.Save(tt.savePath)
			if tt.expectError {
				assert.Error(t, err)
				return
			}
			assert.NoErr(t, err)

			// Load tensor
			loaded, err := LoadTensor(tt.savePath)
			assert.NoErr(t, err)

			// Verify loaded tensor matches original
			assert.Equal(t, tensor.Shape, loaded.Shape)
			assert.Equal(t, tensor.dtype, loaded.dtype)
			assert.Equal(t, tt.data, loaded.ToHost().([]float32))

			// Cleanup
			os.Remove(tt.savePath)
		})
	}

	// Test loading from non-existent file
	_, err := LoadTensor("/nonexistent/file.bin")
	assert.Error(t, err)

	// Test loading from invalid file
	invalidFile := "/tmp/invalid.bin"
	err = os.WriteFile(invalidFile, []byte("invalid data"), 0644)
	assert.NoErr(t, err)
	_, err = LoadTensor(invalidFile)
	assert.Error(t, err)
	os.Remove(invalidFile)

	// Test saving unsupported dtype
	tensor := NewTensor(Shape{2, 2}, GGML_TYPE_F64)
	err = tensor.Save("/tmp/unsupported.bin")
	assert.Error(t, err)
}
func TestTensorGetElem(t *testing.T) {
	tests := []struct {
		shape    Shape
		data     []float32
		index    int
		expected float32
	}{
		{Shape{2, 3}, []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 0, 1.0},
		{Shape{2, 3}, []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 5, 6.0},
		{Shape{2, 3}, []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, -1, 6.0},
		{Shape{2, 3}, []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, -6, 1.0},
	}

	for _, tt := range tests {
		a, err := MakeTensorErr(tt.shape, tt.data)
		assert.NoErr(t, err)
		result := a.GetElem(tt.index)
		assert.Equal(t, tt.expected, result)
	}

	assert.Panic(t, func() {
		a, err := MakeTensorErr(Shape{2, 3}, []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
		assert.NoErr(t, err)
		a.GetElem(6)
	})

	assert.Panic(t, func() {
		a, err := MakeTensorErr(Shape{2, 3}, []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
		assert.NoErr(t, err)
		a.GetElem(-7)
	})
}
func TestTensorRowSlice(t *testing.T) {
	tests := []struct {
		shape       Shape
		data        []float32
		start, end  int
		expected    []float32
		expectPanic bool
	}{
		{
			shape:    Shape{4, 3},
			data:     []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			start:    1,
			end:      3,
			expected: []float32{4, 5, 6, 7, 8, 9},
		},
		{
			shape:    Shape{4, 3},
			data:     []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			start:    -3,
			end:      -1,
			expected: []float32{4, 5, 6, 7, 8, 9},
		},
		{
			shape:       Shape{4, 3},
			data:        []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			start:       3,
			end:         1,
			expectPanic: true,
		},
		{
			shape:       Shape{4, 3},
			data:        []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			start:       1,
			end:         5,
			expectPanic: true,
		},
		{
			shape:       Shape{4, 3},
			data:        []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			start:       -5,
			end:         2,
			expectPanic: true,
		},
		{
			shape:       Shape{4, 3},
			data:        []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			start:       1,
			end:         -5,
			expectPanic: true,
		},
		{
			shape:       Shape{3, 3, 3},
			data:        []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
			start:       1,
			end:         2,
			expectPanic: true,
		},
	}

	for _, tt := range tests {
		a, err := MakeTensorErr(tt.shape, tt.data)
		assert.NoErr(t, err)

		if tt.expectPanic {
			assert.Panic(t, func() { a.RowSlice(tt.start, tt.end) })
			continue
		}

		result := a.RowSlice(tt.start, tt.end)
		assert.Equal(t, Shape{tt.end - tt.start, tt.shape[1]}, result.Shape)
		assert.Equal(t, tt.expected, result.ToHost())
	}
}
