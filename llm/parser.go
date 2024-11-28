package llm

import (
	"encoding/binary"
	"fmt"
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

func reverseSlice(slice []int) {
	if len(slice) == 0 || len(slice) == 1 {
		return
	}
	for i, j := 0, len(slice)-1; i < j; i, j = i+1, j-1 {
		slice[i], slice[j] = slice[j], slice[i]
	}
}

const GGML_MAX_DIMS = 4

// GGUFTensorInfo contains information about a tensor in the GGUF file
type GGUFTensorInfo struct {
	// The name of the tensor. It is a standard GGUF string, with the caveat that
	// it must be at most 64 bytes long.
	// Name string // GGUF string is transformed to go string after parsing from file
	// The number of dimensions in the tensor.
	// Currently at most 4, but this may change in the future.
	// NumDims uint32
	// The dimensions of the tensor.
	Dims []int
	// The type of the tensor.
	Type DType
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
	TensorInfos map[string]GGUFTensorInfo

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
	Alignment int64
	Offset    int64 // Offset of 'data' from the start of the file.
	// Size       int64 // Size of 'data' in bytes.
	// TensorData []byte
	Tensors map[string]*Tensor
}

// GGUFParser reads a GGUF file and returns a parsed GGUFFile structure
func GGUFParser(filename string) (*GGUFFile, error) {
	file, err := MmapOpen(filename)
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

	if gguf.Header.TensorCount == 0 {
		file.Close()
		return gguf, nil
	}

	gguf.TensorInfos = make(map[string]GGUFTensorInfo, gguf.Header.TensorCount)
	for i := uint64(0); i < gguf.Header.TensorCount; i++ {
		name, info, err := readTensorInfo(file)
		if err != nil {
			return nil, fmt.Errorf("failed to read tensor info %d: %w", i, err)
		}
		gguf.TensorInfos[name] = info
	}

	gguf.Offset = file.AlignOffset(gguf.Alignment)
	gguf.Tensors = loadTensors(file, gguf)
	return gguf, nil
}

// readGGUFString reads a GGUF string from the file and transforms it to go string
func readGGUFString(file *mmapReader) (string, error) {
	var length uint64
	if err := binary.Read(file, binary.LittleEndian, &length); err != nil {
		return "", fmt.Errorf("failed to read string length: %w", err)
	}

	str := make([]byte, length)
	if err := binary.Read(file, binary.LittleEndian, &str); err != nil {
		return "", fmt.Errorf("failed to read string data: %w", err)
	}

	return string(str), nil
}

// readGGUFValue reads a GGUF value from the file and transforms it to go value
func readGGUFValue(file *mmapReader, valueType GGUFType) (any, error) {
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

func readKV(file *mmapReader) (GGUFKV, error) {
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

func readTensorInfo(file *mmapReader) (string, GGUFTensorInfo, error) {
	var info GGUFTensorInfo

	name, err := readGGUFString(file)
	if err != nil {
		return "", info, fmt.Errorf("failed to read tensor name: %w", err)
	}

	var numDims uint32
	if err := binary.Read(file, binary.LittleEndian, &numDims); err != nil {
		return name, info, fmt.Errorf("failed to read number of dimensions: %w", err)
	}

	if numDims > GGML_MAX_DIMS {
		return name, info, fmt.Errorf("number of dimensions exceeds maximum: %d > %d", numDims, GGML_MAX_DIMS)
	}

	dims := make([]uint64, numDims)
	for i := uint32(0); i < numDims; i++ {
		if err := binary.Read(file, binary.LittleEndian, &dims[i]); err != nil {
			return name, info, fmt.Errorf("failed to read dimension %d: %w", i, err)
		}
		if int(dims[i]) <= 0 {
			return name, info, fmt.Errorf("dimension %d is not a valid uint64: %d", i, dims[i])
		}
		info.Dims = append(info.Dims, int(dims[i]))
	}

	if err := binary.Read(file, binary.LittleEndian, &info.Type); err != nil {
		return name, info, fmt.Errorf("failed to read tensor type: %w", err)
	}

	if err := binary.Read(file, binary.LittleEndian, &info.Offset); err != nil {
		return name, info, fmt.Errorf("failed to read tensor offset: %w", err)
	}
	// GGUF-convert reverses it for GGML, we reverse it back
	reverseSlice(info.Dims)

	return name, info, nil
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
	for name, info := range g.TensorInfos {
		s += fmt.Sprintf("Name: %s, Dims: %v, Quant: %d\n", name, info.Dims, info.Type)
	}
	fmt.Fprint(st, s)
}

var unicodeToByteMap = buildUnicodeToByteMap()

func buildUnicodeToByteMap() map[uint16]byte {
	reverseMap := make(map[uint16]byte, 256)
	// ASCII characters from '!' to '~' is identity-mapped, no need to add them
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
func unicodeToBytes(tokens []string) {
	for idx, str := range tokens {
		data := []byte(str)
		length := len(data)

		processed := make([]byte, 0, length)

		for i := 0; i < length; {
			// Handle two-byte UTF-8 sequence ranging from 0xc2(194) to 0xc5(197)
			if i+1 < length && data[i] >= 194 && data[i] <= 197 {
				charCode := uint16(((uint16(data[i])-194)*64 + uint16(data[i+1])))
				if orig, ok := unicodeToByteMap[charCode]; ok {
					// fmt.Printf("%d, [%d, %d] -> %d\n", idx, data[i], data[i+1], orig)
					processed = append(processed, orig)
				} else {
					panic(fmt.Sprintf("invalid character code: %d at position %d at %d", charCode, i, idx))
				}
				i += 2
			} else {
				processed = append(processed, data[i])
				i++
			}
		}
		// fmt.Printf("%d, %s -> %s\n", idx, str, processed)
		tokens[idx] = string(processed)
	}
}

func (g *GGUFFile) GetTokensMap() map[string]int32 {
	tokens := g.KVs["tokenizer.ggml.tokens"].([]string)
	tm := make(map[string]int32, len(tokens))
	isGPT2 := g.KVs["tokenizer.ggml.model"] == "gpt2"

	if isGPT2 {
		unicodeToBytes(tokens)
	}
	for i, t := range tokens {
		tm[t] = int32(i)
	}
	return tm
}

func (g *GGUFFile) GetTokenizer() *Tokenizer {
	tokens := g.GetTokensMap()
	tok := NewTokenizer(tokens, &Llama3Handler{})
	return tok
}

func loadTensors(file *mmapReader, g *GGUFFile) map[string]*Tensor {
	tensors := make(map[string]*Tensor, len(g.TensorInfos))
	for name, info := range g.TensorInfos {
		// fmt.Printf("name: %s, shape: %v, dtype: %v\n", name, shape, info.Type)
		p, err := file.PointerAt(int64(info.Offset) + g.Offset)
		if err != nil {
			return nil
		}
		t, err := MakeTensorFrom(info.Dims, p, info.Type)
		if err != nil {
			return nil
		}
		tensors[name] = t
	}

	return tensors
}

func (g *GGUFFile) GetConfig() *Config {
	arch := g.KVs["general.architecture"].(string)
	conf := &Config{
		Arch:       arch,
		ContextLen: g.KVs[arch+".context_length"].(uint32),
		NumHidden:  g.KVs[arch+".block_count"].(uint32),
		NumHead:    g.KVs[arch+".attention.head_count"].(uint32),
		EmbedDim:   g.KVs[arch+".embedding_length"].(uint32),
		RopeTheta:  g.KVs[arch+".rope.freq_base"].(float32),
	}
	eps := g.KVs[arch+".attention.layer_norm_rms_epsilon"]
	if eps == nil {
		eps = g.KVs[arch+".attention.layer_norm_epsilon"]
	}
	conf.NormEpsilon = eps.(float32)

	conf.HeadDim = conf.EmbedDim / conf.NumHead
	conf.NumKVHead = conf.NumHead
	if kvh := g.KVs[arch+".attention.head_count_kv"]; kvh != nil {
		conf.NumKVHead = kvh.(uint32)
	}
	return conf
}
