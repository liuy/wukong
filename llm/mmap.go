package llm

import (
	"fmt"
	"io"
	"os"
	"runtime"
	"syscall"
	"unsafe"
)

// mmapReader reads a memory-mapped file.
//
// Like any io.Reader, clients can execute parallel ReadAt calls, but it is
// not safe to call Close and reading methods concurrently.
type mmapReader struct {
	data   []byte
	offset int64
}

// Close closes the reader.
func (r *mmapReader) Close() error {
	if r.data == nil {
		return nil
	} else if len(r.data) == 0 {
		r.data = nil
		return nil
	}
	data := r.data
	r.data = nil
	runtime.SetFinalizer(r, nil)
	return syscall.Munmap(data)
}

// AlignOffset aligns the offset to the specified alignment.
func (r *mmapReader) AlignOffset(alignment int64) int64 {
	r.offset += (alignment - (r.offset % alignment)) % alignment
	return r.offset
}

// Len returns the length of the underlying memory-mapped file.
func (r *mmapReader) Len() int {
	return len(r.data)
}

// Read reads data into p and returns the number of bytes read.
func (r *mmapReader) Read(p []byte) (int, error) {
	n, err := r.ReadAt(p, r.offset)
	r.offset += int64(n)
	return n, err
}

// // Seek sets the offset for the next Read or Write.
// func (r *mmapReader) Seek(offset int64, whence int) (int64, error) {
// 	switch whence {
// 	case io.SeekStart:
// 		r.offset = offset
// 	case io.SeekCurrent:
// 		r.offset += offset
// 	case io.SeekEnd:
// 		r.offset = int64(r.Len()) + offset
// 	default:
// 		return 0, fmt.Errorf("invalid whence, should be io.SeekStart, io.SeekCurrent or io.SeekEnd")
// 	}
// 	if r.offset < 0 {
// 		return 0, fmt.Errorf("invalid offset causes negative position, %d", r.offset)
// 	}
// 	return r.offset, nil
// }

// ReadAt reads data into p at offset off in the underlying input source.
func (r *mmapReader) ReadAt(p []byte, off int64) (int, error) {
	if r.data == nil {
		return 0, fmt.Errorf("mmap: reader is closed")
	}
	if off < 0 || int64(len(r.data)) < off {
		return 0, fmt.Errorf("mmap: invalid ReadAt offset %d", off)
	}
	n := copy(p, r.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

// Open memory-maps the named file for reading.
func MmapOpen(filename string) (*mmapReader, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}

	size := fi.Size()
	if size <= 0 {
		return nil, fmt.Errorf("mmap: file %q size must be greater than 0", filename)
	}
	// Uncomment to support 32-bit systems if we ever need to
	//
	// size != int64(int(size)) happens if the file is too large to be
	// addressable in a single allocation. This is only possible on 32-bit
	// systems, and even then it requires a huge file.
	//
	// The check is a little subtle: int64(int(size)) is not the same as
	// just size, because size is an int64 and int64(int(size)) is an int64
	// created by truncating an int. If size is too large, the truncation
	// will wrap around, and int64(int(size)) will be smaller than size.
	//
	// The effect of this check is to prevent a 32-bit system from trying
	// to allocate a huge block of memory, which would trigger the OOM
	// killer. Instead, we return an error.
	// if size != int64(int(size)) {
	// 	return nil, fmt.Errorf("mmap: file %q is too large", filename)
	// }

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}
	r := &mmapReader{data, 0}
	runtime.SetFinalizer(r, (*mmapReader).Close)
	return r, nil
}

// PointerAt returns a pointer to the byte at the specified offset.
func (r *mmapReader) PointerAt(offset int64) (unsafe.Pointer, error) {
	if r.data == nil {
		return nil, fmt.Errorf("mmap: reader is closed")
	}
	if offset < 0 || int64(len(r.data)) < offset {
		return nil, fmt.Errorf("mmap: invalid offset %d", offset)
	}
	return unsafe.Pointer(&r.data[offset]), nil
}
