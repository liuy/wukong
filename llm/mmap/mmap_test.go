package mmap

import (
	"io"
	"os"
	"testing"

	"github.com/liuy/wukong/assert"
)

func TestReader(t *testing.T) {
	content := []byte("Hello, World! This is a test file.")
	tmpfile, err := os.CreateTemp("", "est")
	assert.NoErr(t, err)
	defer os.Remove(tmpfile.Name())

	_, err = tmpfile.Write(content)
	assert.NoErr(t, err)
	err = tmpfile.Close()
	assert.NoErr(t, err)

	t.Run("Open", func(t *testing.T) {
		reader, err := Open(tmpfile.Name())
		assert.NoErr(t, err)
		defer reader.Close()

		assert.Equal(t, len(content), reader.Len())

		zeroSizeFile, err := os.CreateTemp("", "est_zero_size")
		assert.NoErr(t, err)
		defer os.Remove(zeroSizeFile.Name())
		err = zeroSizeFile.Close()
		assert.NoErr(t, err)
		_, err = Open(zeroSizeFile.Name())
		assert.Error(t, err)
	})

	t.Run("Read", func(t *testing.T) {
		reader, err := Open(tmpfile.Name())
		assert.NoErr(t, err)
		defer reader.Close()

		buf := make([]byte, 5)
		n, err := reader.Read(buf)
		assert.NoErr(t, err)
		assert.Equal(t, 5, n)
		assert.Equal(t, "Hello", string(buf))
	})

	t.Run("ReadAt", func(t *testing.T) {
		reader, err := Open(tmpfile.Name())
		assert.NoErr(t, err)
		defer reader.Close()

		buf := make([]byte, 5)
		n, err := reader.ReadAt(buf, 7)
		assert.NoErr(t, err)
		assert.Equal(t, 5, n)
		assert.Equal(t, "World", string(buf))
	})

	t.Run("AlignOffset", func(t *testing.T) {
		reader, err := Open(tmpfile.Name())
		assert.NoErr(t, err)
		defer reader.Close()

		buf := make([]byte, 5)
		n, err := reader.Read(buf)
		assert.NoErr(t, err)
		assert.Equal(t, 5, n)
		aligned := reader.AlignOffset(8)
		assert.Equal(t, int64(8), aligned)
	})

	t.Run("PointerAt", func(t *testing.T) {
		reader, err := Open(tmpfile.Name())
		assert.NoErr(t, err)
		defer reader.Close()

		ptr, err := reader.PointerAt(0)
		assert.NoErr(t, err)
		assert.NotNil(t, ptr)
		b := *(*byte)(ptr)
		assert.Equal(t, content[0], b)

		ptr, err = reader.PointerAt(7)
		assert.NoErr(t, err)
		assert.NotNil(t, ptr)
		b = *(*byte)(ptr)
		assert.Equal(t, content[7], b)
	})

	t.Run("ErrorCases", func(t *testing.T) {
		// Test reading from closed reader
		reader, err := Open(tmpfile.Name())
		assert.NoErr(t, err)
		reader.Close()

		buf := make([]byte, 5)
		_, err = reader.Read(buf)
		assert.Error(t, err)

		// Test invalid ReadAt offset
		reader, err = Open(tmpfile.Name())
		assert.NoErr(t, err)

		_, err = reader.ReadAt(buf, -1)
		assert.Error(t, err)

		_, err = reader.ReadAt(buf, int64(len(content)+1))
		assert.Error(t, err)

		_, err = reader.PointerAt(-1)
		assert.Error(t, err)

		_, err = reader.PointerAt(int64(len(content) + 1))
		assert.Error(t, err)

		reader.Close()
		_, err = reader.PointerAt(0)
		assert.Error(t, err)
	})

	t.Run("EOF", func(t *testing.T) {
		reader, err := Open(tmpfile.Name())
		assert.NoErr(t, err)
		defer reader.Close()

		buf := make([]byte, 100)
		n, err := reader.Read(buf)
		assert.Equal(t, io.EOF, err)
		assert.Equal(t, len(content), n)
	})

	t.Run("NonExistentFile", func(t *testing.T) {
		_, err := Open("nonexistent.file")
		assert.Error(t, err)
	})
}
