// Package gguf reads just enough GGUF metadata to size the KV cache: the attention
// dimensions that determine bytes-per-token (layers x kv-heads x head-dim), so the
// "guess max context" estimate is real, not a guess off file size.
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

var le = binary.LittleEndian

// Params are the KV-cache-relevant model dimensions.
type Params struct {
	Arch     string
	Layers   uint32 // block_count
	KVHeads  uint32 // attention.head_count_kv
	HeadDim  uint32 // attention.key_length (or embedding_length/head_count)
	TrainCtx uint32 // context_length
}

// KVBytesPerToken returns the KV-cache bytes per token for a cache element size
// (q8_0 ~1.06, f16 2.0, q4_0 ~0.56). 0 if dims are unknown.
func (p Params) KVBytesPerToken(bytesPerElem float64) float64 {
	if p.Layers == 0 || p.KVHeads == 0 || p.HeadDim == 0 {
		return 0
	}
	// K and V each: layers * kv_heads * head_dim elements
	return 2 * float64(p.Layers) * float64(p.KVHeads) * float64(p.HeadDim) * bytesPerElem
}

// Read parses the GGUF metadata header of a model file.
func Read(path string) (Params, error) {
	f, err := os.Open(path)
	if err != nil {
		return Params{}, err
	}
	defer f.Close()
	r := io.LimitReader(f, 8<<20) // metadata lives in the first few MB

	var magic, version uint32
	var tensorCount, kvCount uint64
	if err := binary.Read(r, le, &magic); err != nil || magic != 0x46554747 {
		return Params{}, fmt.Errorf("not a gguf file")
	}
	binary.Read(r, le, &version)
	binary.Read(r, le, &tensorCount)
	if err := binary.Read(r, le, &kvCount); err != nil {
		return Params{}, err
	}
	meta := map[string]any{}
	for i := uint64(0); i < kvCount; i++ {
		key, err := readString(r)
		if err != nil {
			break
		}
		val, err := readValue(r)
		if err != nil {
			break
		}
		meta[key] = val
	}
	arch, _ := meta["general.architecture"].(string)
	p := Params{Arch: arch}
	p.Layers = u32(meta[arch+".block_count"])
	p.KVHeads = u32(meta[arch+".attention.head_count_kv"])
	p.HeadDim = u32(meta[arch+".attention.key_length"])
	p.TrainCtx = u32(meta[arch+".context_length"])
	if p.HeadDim == 0 { // derive from embedding / head_count
		if emb, heads := u32(meta[arch+".embedding_length"]), u32(meta[arch+".attention.head_count"]); heads > 0 {
			p.HeadDim = emb / heads
		}
	}
	if p.KVHeads == 0 { // models without GQA: kv heads == attention heads
		p.KVHeads = u32(meta[arch+".attention.head_count"])
	}
	return p, nil
}

func readValue(r io.Reader) (any, error) {
	var t uint32
	if err := binary.Read(r, le, &t); err != nil {
		return nil, err
	}
	return readTyped(r, t)
}

func readTyped(r io.Reader, t uint32) (any, error) {
	switch t {
	case 0:
		var v uint8
		binary.Read(r, le, &v)
		return uint64(v), nil
	case 1:
		var v int8
		binary.Read(r, le, &v)
		return int64(v), nil
	case 2:
		var v uint16
		binary.Read(r, le, &v)
		return uint64(v), nil
	case 3:
		var v int16
		binary.Read(r, le, &v)
		return int64(v), nil
	case 4:
		var v uint32
		binary.Read(r, le, &v)
		return uint64(v), nil
	case 5:
		var v int32
		binary.Read(r, le, &v)
		return int64(v), nil
	case 6:
		var v float32
		binary.Read(r, le, &v)
		return v, nil
	case 7:
		var v uint8
		binary.Read(r, le, &v)
		return v != 0, nil
	case 8:
		return readString(r)
	case 10:
		var v uint64
		binary.Read(r, le, &v)
		return v, nil
	case 11:
		var v int64
		binary.Read(r, le, &v)
		return v, nil
	case 12:
		var v float64
		binary.Read(r, le, &v)
		return v, nil
	case 9: // array — read elem type + count, return the first element (e.g. per-layer kv heads)
		var et uint32
		var n uint64
		binary.Read(r, le, &et)
		if err := binary.Read(r, le, &n); err != nil {
			return nil, err
		}
		var first any
		for i := uint64(0); i < n; i++ {
			v, err := readTyped(r, et)
			if err != nil {
				return first, err
			}
			if i == 0 {
				first = v
			}
		}
		return first, nil
	}
	return nil, fmt.Errorf("unknown gguf value type %d", t)
}

func readString(r io.Reader) (string, error) {
	var n uint64
	if err := binary.Read(r, le, &n); err != nil {
		return "", err
	}
	if n > 1<<20 {
		return "", fmt.Errorf("gguf string too long (%d)", n)
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(r, b); err != nil {
		return "", err
	}
	return string(b), nil
}

func u32(v any) uint32 {
	switch x := v.(type) {
	case uint64:
		return uint32(x)
	case int64:
		return uint32(x)
	case uint32:
		return x
	}
	return 0
}
