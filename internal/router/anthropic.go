package router

// The Anthropic door — /v1/messages through the router, and the one
// incompatibility that makes it need help.
//
// llama-server speaks the Anthropic Messages API natively at the slot; the
// router simply never forwarded it, so the real Claude Code harness could only
// reach the rig through a hand-run compatibility shim. Routing it by the
// body's `model` like every other proxied endpoint gets the federation path
// for free.
//
// THE INCOMPATIBILITY: Claude Code 2.x sends mid-conversation role:"system"
// messages (its "Available agent types" turn is the one that caught this).
// Anthropic's own API accepts a system turn anywhere; llama-server maps them
// positionally into the chat template, and Qwen's strict template raises
//
//     System message must be at the beginning
//
// as a 500. foldSystemTurns is a port of the shim's fix, and the shim is the
// spec (it is kept with the operator's private tooling): fold each system
// message's blocks into the nearest PRECEDING user turn, else the next one,
// else append them as a user turn — order and content preserved.
//
// A SCAR WORTH CARRYING FORWARD, from the shim's own comments: the trivial
// shapes (multi-block system, tools) all passed. Only the real captured
// payload showed the true trigger. So the unit cases below are the shapes
// that actually broke, plus their inverses — not the ones convenient to
// write.

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// ordObj is a JSON object that remembers its key order.
//
// encoding/json round-trips objects through map[string]…, which re-sorts keys
// alphabetically — so a naive decode/mutate/encode would rewrite parts of the
// payload this function is supposed to leave alone. The promise is "messages
// changed, everything else untouched", and that promise is worth keeping
// literally rather than approximately.
type ordObj []ordField

type ordField struct {
	Key string
	Val json.RawMessage
}

func parseOrdered(raw []byte) (ordObj, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, fmt.Errorf("not a JSON object")
	}
	var out ordObj
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, err
		}
		key, ok := kt.(string)
		if !ok {
			return nil, fmt.Errorf("non-string object key")
		}
		var val json.RawMessage
		if err := dec.Decode(&val); err != nil {
			return nil, err
		}
		out = append(out, ordField{Key: key, Val: val})
	}
	if _, err := dec.Token(); err != nil { // closing '}'
		return nil, err
	}
	return out, nil
}

func (o ordObj) get(key string) (json.RawMessage, bool) {
	for _, f := range o {
		if f.Key == key {
			return f.Val, true
		}
	}
	return nil, false
}

// set replaces a key's value IN PLACE (keeping its position), or appends.
func (o ordObj) set(key string, val json.RawMessage) ordObj {
	for i := range o {
		if o[i].Key == key {
			o[i].Val = val
			return o
		}
	}
	return append(o, ordField{Key: key, Val: val})
}

func (o ordObj) marshal() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, f := range o {
		if i > 0 {
			b.WriteByte(',')
		}
		k, err := json.Marshal(f.Key)
		if err != nil {
			return nil, err
		}
		b.Write(k)
		b.WriteByte(':')
		b.Write(f.Val)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// roleOf returns a message's role, or "" if absent/not a string.
func roleOf(m ordObj) string {
	raw, ok := m.get("role")
	if !ok {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return ""
	}
	return s
}

// blocksOf returns a message's content as Anthropic content blocks. A bare
// string becomes one text block — the same normalisation the shim does, and
// the reason a folded-into target's content changes shape from string to
// list (semantically identical to the API, and unavoidable when appending).
func blocksOf(m ordObj) ([]json.RawMessage, error) {
	raw, ok := m.get("content")
	if !ok {
		return nil, nil
	}
	var arr []json.RawMessage
	if err := json.Unmarshal(raw, &arr); err == nil {
		return arr, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, fmt.Errorf("content is neither a block list nor a string")
	}
	blk, err := json.Marshal(map[string]string{"type": "text", "text": s})
	if err != nil {
		return nil, err
	}
	return []json.RawMessage{blk}, nil
}

func packBlocks(blocks []json.RawMessage) (json.RawMessage, error) {
	var b bytes.Buffer
	b.WriteByte('[')
	for i, blk := range blocks {
		if i > 0 {
			b.WriteByte(',')
		}
		b.Write(blk)
	}
	b.WriteByte(']')
	return json.RawMessage(b.Bytes()), nil
}

// foldSystemTurns folds mid-conversation role:"system" messages into user
// turns so a strict chat template accepts the conversation.
//
// Returns the rewritten body and how many system messages were folded. When
// there is NOTHING to fold it returns the caller's original bytes untouched —
// not a re-serialisation of them — so the overwhelmingly common request is
// forwarded exactly as it arrived.
//
// Never fatal: a body this cannot parse is returned unchanged with the error,
// and the caller forwards the original. The upstream's own error is a better
// answer than one invented here.
func foldSystemTurns(body []byte) ([]byte, int, error) {
	doc, err := parseOrdered(body)
	if err != nil {
		return body, 0, err
	}
	rawMsgs, ok := doc.get("messages")
	if !ok {
		return body, 0, nil
	}
	var msgRaw []json.RawMessage
	if err := json.Unmarshal(rawMsgs, &msgRaw); err != nil {
		return body, 0, err
	}

	msgs := make([]ordObj, 0, len(msgRaw))
	anySystem := false
	for _, r := range msgRaw {
		m, err := parseOrdered(r)
		if err != nil {
			return body, 0, err
		}
		if roleOf(m) == "system" {
			anySystem = true
		}
		msgs = append(msgs, m)
	}
	if !anySystem {
		return body, 0, nil // byte-identical passthrough
	}

	var out []ordObj
	var pending []json.RawMessage
	folded := 0

	for _, m := range msgs {
		if roleOf(m) == "system" {
			folded++
			blks, err := blocksOf(m)
			if err != nil {
				return body, 0, err
			}
			// nearest PRECEDING user turn
			tgt := -1
			for i := len(out) - 1; i >= 0; i-- {
				if roleOf(out[i]) == "user" {
					tgt = i
					break
				}
			}
			if tgt >= 0 {
				cur, err := blocksOf(out[tgt])
				if err != nil {
					return body, 0, err
				}
				packed, err := packBlocks(append(cur, blks...))
				if err != nil {
					return body, 0, err
				}
				out[tgt] = out[tgt].set("content", packed)
			} else {
				pending = append(pending, blks...) // …else the NEXT user turn
			}
			continue
		}
		if len(pending) > 0 && roleOf(m) == "user" {
			cur, err := blocksOf(m)
			if err != nil {
				return body, 0, err
			}
			packed, err := packBlocks(append(pending, cur...))
			if err != nil {
				return body, 0, err
			}
			m = append(ordObj(nil), m...) // copy before mutating
			m = m.set("content", packed)
			pending = nil
		}
		out = append(out, m)
	}
	if len(pending) > 0 { // …else append as a user turn of its own
		packed, err := packBlocks(pending)
		if err != nil {
			return body, 0, err
		}
		out = append(out, ordObj{
			{Key: "role", Val: json.RawMessage(`"user"`)},
			{Key: "content", Val: packed},
		})
	}

	var buf bytes.Buffer
	buf.WriteByte('[')
	for i, m := range out {
		if i > 0 {
			buf.WriteByte(',')
		}
		b, err := m.marshal()
		if err != nil {
			return body, 0, err
		}
		buf.Write(b)
	}
	buf.WriteByte(']')

	newBody, err := doc.set("messages", json.RawMessage(buf.Bytes())).marshal()
	if err != nil {
		return body, 0, err
	}
	return newBody, folded, nil
}
