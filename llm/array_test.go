package llm

import (
	"fmt"
	"math"
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
		{Shape{4, 2, 2}, RandFloatSlice(16), true},
		{Shape{2, 3}, RandFloatSlice(6), true},
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
			aData: []float32{1, 2, 3, 4, 5, 6}, bData: []float32{7, 8, 9, 10}, biasData: nil,
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
