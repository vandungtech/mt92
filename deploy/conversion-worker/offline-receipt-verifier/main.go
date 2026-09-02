// Offline worker-receipt signature verifier.
//
// Invoked by training/verify_code_conversion_export.py's CommandSignatureVerifier as:
//
//	<verifier> verify --key K --signature S --message M --scheme SCHEME --key-id KEYID
//
// Every path is a /proc/self/fd/N reference to a descriptor the caller holds open;
// opening that path yields a fresh description at offset 0.
//
// Exit 0  signature is valid for the message under the trusted key
// Exit 1  signature is invalid, or the key/scheme/key-id do not match
// Exit 2  malformed input or I/O failure
//
// Built with CGO_ENABLED=0 so the result is a static ELF64 with no PT_INTERP and no
// PT_DYNAMIC, as _validate_static_verifier_elf requires. Ed25519 comes from the Go
// standard library; there is no third-party cryptographic dependency to review.
package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"os"
)

const (
	schemeName    = "ed25519"
	maxMessage    = 64 << 20 // 64 MiB, matching the verifier's own read ceiling
	maxSmallField = 4 << 10
)

func fail(code int, format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(code)
}

// readBounded opens path fresh (offset 0) and reads at most limit+1 bytes so an
// oversized input is detected rather than silently truncated.
func readBounded(path string, limit int64) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(raw)) > limit {
		return nil, fmt.Errorf("%s exceeds %d bytes", path, limit)
	}
	return raw, nil
}

// decodeHex accepts exactly want bytes of lowercase hex with at most one trailing newline.
func decodeHex(raw []byte, want int, label string) ([]byte, error) {
	trimmed := bytes.TrimSuffix(raw, []byte("\n"))
	if len(trimmed) != want*2 {
		return nil, fmt.Errorf("%s must be %d lowercase hex characters", label, want*2)
	}
	out := make([]byte, want)
	if _, err := hex.Decode(out, trimmed); err != nil {
		return nil, fmt.Errorf("%s is not valid hex", label)
	}
	if !bytes.Equal(bytes.ToLower(trimmed), trimmed) {
		return nil, fmt.Errorf("%s must be lowercase", label)
	}
	return out, nil
}

func main() {
	if len(os.Args) < 2 || os.Args[1] != "verify" {
		fail(2, "usage: verifier verify --key K --signature S --message M --scheme S --key-id K")
	}
	fs := flag.NewFlagSet("verify", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	keyPath := fs.String("key", "", "trusted public key file")
	sigPath := fs.String("signature", "", "detached signature file")
	msgPath := fs.String("message", "", "signed message file")
	scheme := fs.String("scheme", "", "signature scheme")
	keyID := fs.String("key-id", "", "expected key identifier")
	if err := fs.Parse(os.Args[2:]); err != nil {
		fail(2, "argument error: %v", err)
	}
	if *keyPath == "" || *sigPath == "" || *msgPath == "" || *scheme == "" || *keyID == "" {
		fail(2, "every argument is required")
	}
	if *scheme != schemeName {
		fail(1, "unsupported scheme %q", *scheme)
	}

	keyRaw, err := readBounded(*keyPath, maxSmallField)
	if err != nil {
		fail(2, "cannot read key: %v", err)
	}
	pub, err := decodeHex(keyRaw, ed25519.PublicKeySize, "public key")
	if err != nil {
		fail(2, "%v", err)
	}
	// The key id binds the spec's declared identity to the key actually supplied.
	sum := sha256.Sum256(pub)
	if want := hex.EncodeToString(sum[:]); want != *keyID {
		fail(1, "key id mismatch")
	}

	sigRaw, err := readBounded(*sigPath, maxSmallField)
	if err != nil {
		fail(2, "cannot read signature: %v", err)
	}
	sig, err := decodeHex(sigRaw, ed25519.SignatureSize, "signature")
	if err != nil {
		fail(2, "%v", err)
	}

	msg, err := readBounded(*msgPath, maxMessage)
	if err != nil {
		fail(2, "cannot read message: %v", err)
	}

	if !ed25519.Verify(ed25519.PublicKey(pub), msg, sig) {
		fail(1, "signature does not verify")
	}
	os.Exit(0)
}
