package assert

import (
	"reflect"
	"runtime"
)

// TB is an interface implemented by both *testing.T and *testing.B.
type TB interface {
	Helper()
	Errorf(format string, args ...interface{})
}

// True asserts that the condition is true.
func True(t TB, condition bool) {
	t.Helper()
	if !condition {
		_, file, line, _ := runtime.Caller(1)
		t.Errorf("\n%s:%d\nExpected: true\nGot: false", file, line)
	}
}

// False asserts that the condition is false.
func False(t TB, condition bool) {
	t.Helper()
	if condition {
		_, file, line, _ := runtime.Caller(1)
		t.Errorf("\n%s:%d\nExpected: false\nGot: true", file, line)
	}
}

// Error asserts that the error is not nil.
func Error(t TB, err error) {
	t.Helper()
	if err == nil {
		_, file, line, _ := runtime.Caller(1)
		t.Errorf("\n%s:%d\nExpected an error\nGot: nil", file, line)
	}
}

// NoErr asserts that the error is nil.
func NoErr(t TB, err error) {
	t.Helper()
	if err != nil {
		_, file, line, _ := runtime.Caller(1)
		t.Errorf("\n%s:%d\nExpected no error\nGot: %v", file, line, err)
	}
}

// Nil asserts that the given value is nil.
func Nil(t TB, data any) {
	t.Helper()
	if !isNil(data) {
		_, file, line, _ := runtime.Caller(1)
		t.Errorf("\n%s:%d\nExpected value to be nil\nGot: %#v", file, line, data)
	}
}

// NotNil asserts that the given value is not nil.
func NotNil(t TB, data any) {
	t.Helper()
	if isNil(data) {
		_, file, line, _ := runtime.Caller(1)
		t.Errorf("\n%s:%d\nExpected value to not be nil\nGot: nil", file, line)
	}
}

// Near asserts that two float32 values are nearly equal.
func Near(t TB, expected, actual, tolerance float32) {
	t.Helper()
	if actual < expected-tolerance || actual > expected+tolerance {
		_, file, line, _ := runtime.Caller(1)
		t.Errorf("\n%s:%d\nExpected: %f ± %f\nGot: %f", file, line, expected, tolerance, actual)
	}
}

// SliceNear asserts that two float32 slices are nearly equal.
func SliceNear(t TB, expected, actual []float32, tolerance float32) {
	t.Helper()
	if len(expected) != len(actual) {
		_, file, line, _ := runtime.Caller(1)
		t.Errorf("\n%s:%d\nExpected: %v\nGot: %v", file, line, expected, actual)
		return
	}

	for i := range expected {
		if actual[i] < expected[i]-tolerance || actual[i] > expected[i]+tolerance {
			_, file, line, _ := runtime.Caller(1)
			t.Errorf("\n%s:%d\nExpected: %v ± %f\nGot: %v", file, line, expected, tolerance, actual)
			return
		}
	}
}

// Equal asserts that two values are equal.
func Equal(t TB, expected, actual any) {
	t.Helper()
	if !reflect.DeepEqual(expected, actual) {
		_, file, line, _ := runtime.Caller(1)
		t.Errorf("\n%s:%d\nExpected:\n%#v\nGot:\n%#v", file, line, expected, actual)
	}
}

// isNil is a helper function that checks if a value is nil.
func isNil(v any) bool {
	if v == nil {
		return true
	}

	val := reflect.ValueOf(v)
	switch val.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice, reflect.UnsafePointer:
		return val.IsNil()
	default:
		return false
	}
}

// Panic asserts that the function panics.
func Panic(t TB, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			_, file, line, _ := runtime.Caller(1)
			t.Errorf("\n%s:%d\nExpected panic\nGot: none", file, line)
		}
	}()
	f()
}
