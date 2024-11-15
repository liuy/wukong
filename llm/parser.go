package llm

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// GGMLType represents the type of tensor data
type GGMLType uint32

const (
	GGML_TYPE_F32      GGMLType = iota // 0
	GGML_TYPE_F16                      // 1
	GGML_TYPE_Q4_0                     // 2
	GGML_TYPE_Q4_1                     // 3
	GGML_TYPE_Q4_2                     // 4 support has been removed
	GGML_TYPE_Q4_3                     // 5 support has been removed
	GGML_TYPE_Q5_0                     // 6
	GGML_TYPE_Q5_1                     // 7
	GGML_TYPE_Q8_0                     // 8
	GGML_TYPE_Q8_1                     // 9
	GGML_TYPE_Q2_K                     // 10
	GGML_TYPE_Q3_K                     // 11
	GGML_TYPE_Q4_K                     // 12
	GGML_TYPE_Q5_K                     // 13
	GGML_TYPE_Q6_K                     // 14
	GGML_TYPE_Q8_K                     // 15
	GGML_TYPE_IQ2_XXS                  // 16
	GGML_TYPE_IQ2_XS                   // 17
	GGML_TYPE_IQ3_XXS                  // 18
	GGML_TYPE_IQ1_S                    // 19
	GGML_TYPE_IQ4_NL                   // 20
	GGML_TYPE_IQ3_S                    // 21
	GGML_TYPE_IQ2_S                    // 22
	GGML_TYPE_IQ4_XS                   // 23
	GGML_TYPE_I8                       // 24
	GGML_TYPE_I16                      // 25
	GGML_TYPE_I32                      // 26
	GGML_TYPE_I64                      // 27
	GGML_TYPE_F64                      // 28
	GGML_TYPE_IQ1_M                    // 29
	GGML_TYPE_BF16                     // 30
	GGML_TYPE_Q4_0_4_4                 // 31
	GGML_TYPE_Q4_0_4_8                 // 32
	GGML_TYPE_Q4_0_8_8                 // 33
	GGML_TYPE_TQ1_0                    // 34
	GGML_TYPE_TQ2_0                    // 35
	GGML_TYPE_COUNT
)

// GGUFType represents the type of metadata value
type GGUFType uint32

const (
	GGUF_TYPE_UINT8   GGUFType = iota // 0 The value is a 8-bit unsigned integer.
	GGUF_TYPE_INT8                    // 1 The value is a 8-bit signed integer.
	GGUF_TYPE_UINT16                  // 2 The value is a 16-bit unsigned little-endian integer.
	GGUF_TYPE_INT16                   // 3 The value is a 16-bit signed little-endian integer.
	GGUF_TYPE_UINT32                  // 4 The value is a 32-bit unsigned little-endian integer.
	GGUF_TYPE_INT32                   // 5 The value is a 32-bit signed little-endian integer.
	GGUF_TYPE_FLOAT32                 // 6 The value is a 32-bit IEEE754 floating point number.
	GGUF_TYPE_BOOL                    // 7 The value is a 8-bit boolean. Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
	GGUF_TYPE_STRING                  // 8 The value is a UTF-8 non-null-terminated string, with length prepended.
	GGUF_TYPE_ARRAY                   // 9 The value is an array of other values, with the length and type prepended. Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
	GGUF_TYPE_UINT64                  // 10 The value is a 64-bit unsigned little-endian integer.
	GGUF_TYPE_INT64                   // 11 The value is a 64-bit signed little-endian integer.
	GGUF_TYPE_FLOAT64                 // 12 The value is a 64-bit IEEE754 floating point number.
	GGUF_TYPE_COUNT
)

// GGUFString represents a string in metadata
// On disk: |N|B|B|B|...| where N is the length of the string, and D is the byte of the string
// In memory: string(BBB...)
// type GGUFString struct {
// 	Len uint64
// 	Str []byte
// }

// GGUFArray represents an array in metadata
// On disk: |T|N|D|D|D|...| where T is the type of the array, N is the number of elements, and D is the data
// in memory: []T{D1, D2, D3...}
// type GGUFArray struct {
// 	Type GGUFType
// 	Num  uint64 // Number of elements, not bytes
// 	Data any    // GGUF array data is transformed to go slice after parsing from file
// }

// GGUFKV represents a key-value pair in metadata
// The key of the metadata. It is a standard GGUF string, with the following caveats:
// - It must be a valid ASCII string.
// - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
// - It must be at most 2^16-1/65535 bytes long.
// Any keys that do not follow these rules are invalid.
type GGUFKV struct {
	Key   string // GGUF string is transformed to go string after parsing from file
	Value any    // GGUF value is transformed to go value after parsing from file
}

// GGUFHeader represents the header of a GGUF file
type GGUFHeader struct {
	// Magic number to announce that this is a GGUF file.
	// Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
	// Your executor might do little-endian byte order, so it might be
	// check for 0x46554747 and letting the endianness cancel out.
	// Consider being *very* explicit about the byte order here.
	Magic [4]byte
	// The version of the format implemented.
	// Must be `3` for version described in this spec, which introduces big-endian support.
	//
	// This version should only be increased for structural changes to the format.
	// Changes that do not affect the structure of the file should instead update the metadata
	// to signify the change.
	Version uint32
	// The number of tensors in the file.
	// This is explicit, instead of being included in the metadata, to ensure it is always present
	// for loading the tensors.
	TensorCount uint64
	// The number of metadata key-value pairs.
	KVCount uint64
}

// AlignOffset aligns an offset to the specified alignment
func AlignOffset(offset int64, alignment int64) int64 {
	return offset + (alignment-(offset%alignment))%alignment
}

const GGML_MAX_DIMS = 4

// GGUFTensorInfo contains information about a tensor in the GGUF file
type GGUFTensorInfo struct {
	// The name of the tensor. It is a standard GGUF string, with the caveat that
	// it must be at most 64 bytes long.
	Name string // GGUF string is transformed to go string after parsing from file
	// The number of dimensions in the tensor.
	// Currently at most 4, but this may change in the future.
	NumDims uint32
	// The dimensions of the tensor.
	Dims [GGML_MAX_DIMS]uint64
	// The type of the tensor.
	Type GGMLType
	// The offset of the tensor's data in this file in bytes.
	//
	// This offset is relative to `tensor_data`, not to the start
	// of the file, to make it easier for writers to write the file.
	// Readers should consider exposing this offset relative to the
	// file to make it easier to read the data.
	//
	// Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.
	Offset uint64
}

const GGUF_DEFAULT_ALIGNMENT = 32

// GGUFFile represents a complete GGUF file structure
type GGUFFile struct {
	// The header of the file.
	Header GGUFHeader

	// The metadata key-value pairs.
	KVs map[string]any

	// Tensor infos, which can be used to locate the tensor data.
	TensorInfos []GGUFTensorInfo

	// Tensor data.
	//
	// This is arbitrary binary data corresponding to the weights of the model. This data should be close
	// or identical to the data in the original model file, but may be different due to quantization or
	// other optimizations for inference. Any such deviations should be recorded in the metadata or as
	// part of the architecture definition.
	//
	// Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
	// The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
	// should be padded to `ALIGNMENT` bytes.
	Alignment  int64
	Offset     int64 // Offset of 'data' from the start of the file.
	Size       int64 // Size of 'data' in bytes.
	TensorData []byte
}

// GGUFParser reads a GGUF file and returns a parsed GGUFFile structure
func GGUFParser(filename string) (*GGUFFile, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	gguf := &GGUFFile{
		Alignment: GGUF_DEFAULT_ALIGNMENT,
	}

	if err := binary.Read(file, binary.LittleEndian, &gguf.Header); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}
	if string(gguf.Header.Magic[:]) != "GGUF" {
		return nil, fmt.Errorf("invalid magic number: expected 'GGUF', got '%s'", string(gguf.Header.Magic[:]))
	}
	if gguf.Header.Version != 3 {
		return nil, fmt.Errorf("unsupported version: expected 3, got %d", gguf.Header.Version)
	}

	if gguf.Header.KVCount != 0 {
		gguf.KVs = make(map[string]any, gguf.Header.KVCount)
		for i := uint64(0); i < gguf.Header.KVCount; i++ {
			kv, err := readKV(file)
			if err != nil {
				return nil, fmt.Errorf("failed to read metadata KV pair %d: %w", i, err)
			}
			gguf.KVs[kv.Key] = kv.Value
		}
	}

	if gguf.Header.TensorCount != 0 {
		gguf.TensorInfos = make([]GGUFTensorInfo, gguf.Header.TensorCount)
		for i := uint64(0); i < gguf.Header.TensorCount; i++ {
			info, err := readTensorInfo(file)
			if err != nil {
				return nil, fmt.Errorf("failed to read tensor info %d: %w", i, err)
			}
			gguf.TensorInfos[i] = info
		}

		gguf.Offset, err = file.Seek(0, io.SeekCurrent)
		if err != nil {
			return nil, fmt.Errorf("failed to get tensor data offset: %w", err)
		}

		gguf.Offset = AlignOffset(gguf.Offset, gguf.Alignment)
		if _, err := file.Seek(gguf.Offset, io.SeekStart); err != nil {
			return nil, fmt.Errorf("failed to seek to aligned tensor data offset: %w", err)
		}

		fileInfo, err := file.Stat()
		if err != nil {
			return nil, fmt.Errorf("failed to get file info: %w", err)
		}

		gguf.Size = fileInfo.Size() - gguf.Offset

		gguf.TensorData = make([]byte, gguf.Size)
		if _, err := file.Read(gguf.TensorData); err != nil {
			return nil, fmt.Errorf("failed to read tensor data: %w", err)
		}
	}

	return gguf, nil
}

// readGGUFString reads a GGUF string from the file and transforms it to go string
func readGGUFString(file *os.File) (string, error) {
	var length uint64
	if err := binary.Read(file, binary.LittleEndian, &length); err != nil {
		return "", fmt.Errorf("failed to read string length: %w", err)
	}

	str := make([]byte, length)
	if _, err := file.Read(str); err != nil {
		return "", fmt.Errorf("failed to read string data: %w", err)
	}

	return string(str), nil
}

// readGGUFValue reads a GGUF value from the file and transforms it to go value
func readGGUFValue(file *os.File, valueType GGUFType) (any, error) {
	var value any

	switch valueType {
	case GGUF_TYPE_UINT8:
		var v uint8
		if err := binary.Read(file, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		value = v

	case GGUF_TYPE_INT8:
		var v int8
		if err := binary.Read(file, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		value = v

	case GGUF_TYPE_UINT16:
		var v uint16
		if err := binary.Read(file, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		value = v

	case GGUF_TYPE_INT16:
		var v int16
		if err := binary.Read(file, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		value = v

	case GGUF_TYPE_UINT32:
		var v uint32
		if err := binary.Read(file, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		value = v

	case GGUF_TYPE_INT32:
		var v int32
		if err := binary.Read(file, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		value = v

	case GGUF_TYPE_FLOAT32:
		var v float32
		if err := binary.Read(file, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		value = v

	case GGUF_TYPE_BOOL:
		var v uint8
		if err := binary.Read(file, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		if v != 0 && v != 1 {
			return nil, fmt.Errorf("invalid boolean value: %d", v)
		}
		value = v != 0

	case GGUF_TYPE_STRING:
		str, err := readGGUFString(file)
		if err != nil {
			return nil, err
		}
		value = str

	case GGUF_TYPE_ARRAY:
		var tp GGUFType
		var num uint64
		if err := binary.Read(file, binary.LittleEndian, &tp); err != nil {
			return nil, err
		}
		if err := binary.Read(file, binary.LittleEndian, &num); err != nil {
			return nil, err
		}
		switch tp {
		case GGUF_TYPE_STRING:
			data := make([]string, num)
			for i := uint64(0); i < num; i++ {
				str, err := readGGUFString(file)
				if err != nil {
					return nil, err
				}
				data[i] = str
			}
			value = data
		case GGUF_TYPE_ARRAY:
			panic("nested arrays in GGUFKV are not supported")
		default:
			data, err := func(tp GGUFType, num uint64) (any, error) {
				switch tp {
				case GGUF_TYPE_UINT8:
					return make([]uint8, num), nil
				case GGUF_TYPE_INT8:
					return make([]int8, num), nil
				case GGUF_TYPE_UINT16:
					return make([]uint16, num), nil
				case GGUF_TYPE_INT16:
					return make([]int16, num), nil
				case GGUF_TYPE_UINT32:
					return make([]uint32, num), nil
				case GGUF_TYPE_INT32:
					return make([]int32, num), nil
				case GGUF_TYPE_FLOAT32:
					return make([]float32, num), nil
				case GGUF_TYPE_BOOL:
					return make([]bool, num), nil
				case GGUF_TYPE_UINT64:
					return make([]uint64, num), nil
				case GGUF_TYPE_INT64:
					return make([]int64, num), nil
				case GGUF_TYPE_FLOAT64:
					return make([]float64, num), nil
				default:
					return nil, fmt.Errorf("unsupported array type: %d", tp)
				}
			}(tp, num)
			if err != nil {
				return nil, err
			}
			binary.Read(file, binary.LittleEndian, data)
			value = data
		}
	case GGUF_TYPE_UINT64:
		var v uint64
		if err := binary.Read(file, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		value = v

	case GGUF_TYPE_INT64:
		var v int64
		if err := binary.Read(file, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		value = v

	case GGUF_TYPE_FLOAT64:
		var v float64
		if err := binary.Read(file, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		value = v

	default:
		return nil, fmt.Errorf("unsupported value type: %d", valueType)
	}

	return value, nil
}

func readKV(file *os.File) (GGUFKV, error) {
	var kv GGUFKV

	key, err := readGGUFString(file)
	if err != nil {
		return kv, fmt.Errorf("failed to read key: %w", err)
	}
	kv.Key = key

	var valueType GGUFType
	if err := binary.Read(file, binary.LittleEndian, &valueType); err != nil {
		return kv, fmt.Errorf("failed to read value type: %w", err)
	}

	value, err := readGGUFValue(file, valueType)
	if err != nil {
		return kv, fmt.Errorf("failed to read value: %w", err)
	}
	kv.Value = value

	return kv, nil
}

func readTensorInfo(file *os.File) (GGUFTensorInfo, error) {
	var info GGUFTensorInfo

	name, err := readGGUFString(file)
	if err != nil {
		return info, fmt.Errorf("failed to read tensor name: %w", err)
	}
	info.Name = name

	if err := binary.Read(file, binary.LittleEndian, &info.NumDims); err != nil {
		return info, fmt.Errorf("failed to read number of dimensions: %w", err)
	}

	if info.NumDims > GGML_MAX_DIMS {
		return info, fmt.Errorf("number of dimensions exceeds maximum: %d > %d", info.NumDims, GGML_MAX_DIMS)
	}

	for i := uint32(0); i < info.NumDims; i++ {
		if err := binary.Read(file, binary.LittleEndian, &info.Dims[i]); err != nil {
			return info, fmt.Errorf("failed to read dimension %d: %w", i, err)
		}
	}

	if err := binary.Read(file, binary.LittleEndian, &info.Type); err != nil {
		return info, fmt.Errorf("failed to read tensor type: %w", err)
	}

	if err := binary.Read(file, binary.LittleEndian, &info.Offset); err != nil {
		return info, fmt.Errorf("failed to read tensor offset: %w", err)
	}

	return info, nil
}

func (g *GGUFFile) Format(st fmt.State, r rune) {
	var s string
	for k, v := range g.KVs {
		if k == "tokenizer.ggml.tokens" || k == "tokenizer.ggml.scores" || k == "tokenizer.ggml.token_type" ||
			k == "tokenizer.ggml.merges" || k == "tokenizer.chat_template" {
			s += fmt.Sprintf("%s: not printed\n", k)
		} else {
			s += fmt.Sprintf("%s: %v\n", k, v)
		}
	}
	for _, info := range g.TensorInfos {
		s += fmt.Sprintf("%s, Dims[", info.Name)
		for i := uint32(0); i < info.NumDims-1; i++ {
			s += fmt.Sprintf("%d ", info.Dims[i])
		}
		s += fmt.Sprintf("%d], Quant: %d\n", info.Dims[info.NumDims-1], info.Type)
	}
	fmt.Fprint(st, s)
}

var unicodeToByteMap = buildUnicodeToByteMap()

func buildUnicodeToByteMap() map[uint16]byte {
	reverseMap := make(map[uint16]byte, 256)
	// ASCII characters from '!' to '~'
	for b := byte('!'); b <= '~'; b++ {
		reverseMap[uint16(b)] = b
	}
	// Latin-1 Supplement characters from '¡' to '¬' and '®' to 'ÿ'
	for b := uint16(0xA1); b <= 0xFF; b++ {
		if b != 0xAD { // Skip 0xAD
			reverseMap[b] = byte(b)
		}
	}
	// Additional mappings
	n := 0
	for b := 0; b < 256; b++ {
		if b < int('!') ||
			(b > int('~') && b < 0xA1) ||
			b == 0xAD {
			reverseMap[uint16(256+n)] = byte(b)
			n++
		}
	}
	return reverseMap
}

// unicodeToBytes translates encoded bytes back to their original form based on the bytes_to_unicode() mapping
// https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
func unicodeToBytes(tokens []string) error {
	for idx, str := range tokens {
		data := []byte(str)
		length := len(data)

		processed := make([]byte, 0, length)

		for i := 0; i < length; {
			// Fast path for ASCII characters
			if data[i] < 128 {
				if orig, ok := unicodeToByteMap[uint16(data[i])]; ok {
					processed = append(processed, orig)
				} else {
					processed = append(processed, data[i])
				}
				i++
				continue
			}
			// Handle two-byte UTF-8 sequence
			if i+1 < length && data[i] >= 194 && data[i] <= 197 {
				charCode := uint16(((uint16(data[i])-194)*64 + uint16(data[i+1])))
				if orig, ok := unicodeToByteMap[charCode]; ok {
					processed = append(processed, orig)
				} else {
					return fmt.Errorf("invalid character code: %d at position %d in string %d", charCode, i, idx)
				}
				i += 2
			} else {
				// Handle single byte non-ASCII character
				if orig, ok := unicodeToByteMap[uint16(data[i])]; ok {
					processed = append(processed, orig)
				} else {
					processed = append(processed, data[i])
				}
				i++
			}
		}
		tokens[idx] = string(processed)
	}
	return nil
}

func (g *GGUFFile) GetTokensMap() map[string]int {
	tokenList := g.KVs["tokenizer.ggml.tokens"].([]string)
	tokens := make(map[string]int, len(tokenList))
	isGPT2 := g.KVs["tokenizer.ggml.model"] == "gpt2"

	if isGPT2 {
		if err := unicodeToBytes(tokenList); err != nil {
			panic(err)
		}
	}
	for i, token := range tokenList {
		tokens[token] = i
	}
	return tokens
}
