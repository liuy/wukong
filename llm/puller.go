package llm

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/sync/errgroup"
)

// TODO: NEED refactoring this whole file. What a mess.

// MissingPart is used to indicate any part of a name that was "promised" by
// the presence of a separator, but is missing.
//
// The value was chosen because it is deemed unlikely to be set by a user,
// not a valid part name valid when checked by [Name.IsValid], and easy to
// spot in logs.
const MissingPart = "!MISSING!"

// DefaultName returns a name with the default values for the host, namespace,
// and tag parts. The model and digest parts are empty.
func DefaultName() Name {
	return Name{
		Host:      DefaultHost,
		Namespace: DefaultNamespace,
		Tag:       DefaultTag,
	}

}

type Set map[string]struct{}

func (m Set) Add(digest string) {
	m[digest] = struct{}{}
}

func (m Set) Remove(digest string) {
	delete(m, digest)
}

func (m Set) Contains(digest string) bool {
	_, exists := m[digest]
	return exists
}

type Layer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	From      string `json:"from,omitempty"`
	status    string
}

func NewLayer(r io.Reader, mediatype string) (Layer, error) {
	blobs, err := GetBlobsPath("")
	if err != nil {
		return Layer{}, err
	}

	temp, err := os.CreateTemp(blobs, "sha256-")
	if err != nil {
		return Layer{}, err
	}
	defer temp.Close()
	defer os.Remove(temp.Name())

	sha256sum := sha256.New()
	n, err := io.Copy(io.MultiWriter(temp, sha256sum), r)
	if err != nil {
		return Layer{}, err
	}

	if err := temp.Close(); err != nil {
		return Layer{}, err
	}

	digest := fmt.Sprintf("sha256:%x", sha256sum.Sum(nil))
	blob, err := GetBlobsPath(digest)
	if err != nil {
		return Layer{}, err
	}

	status := "using existing layer"
	if _, err := os.Stat(blob); err != nil {
		status = "creating new layer"
		if err := os.Rename(temp.Name(), blob); err != nil {
			return Layer{}, err
		}
		if err := os.Chmod(blob, 0o644); err != nil {
			return Layer{}, err
		}
	}

	return Layer{
		MediaType: mediatype,
		Digest:    digest,
		Size:      n,
		status:    fmt.Sprintf("%s %s", status, digest),
	}, nil
}

func NewLayerFromLayer(digest, mediatype, from string) (Layer, error) {
	if digest == "" {
		return Layer{}, errors.New("creating new layer from layer with empty digest")
	}

	blob, err := GetBlobsPath(digest)
	if err != nil {
		return Layer{}, err
	}

	fi, err := os.Stat(blob)
	if err != nil {
		return Layer{}, err
	}

	return Layer{
		MediaType: mediatype,
		Digest:    digest,
		Size:      fi.Size(),
		From:      from,
		status:    fmt.Sprintf("using existing layer %s", digest),
	}, nil
}

func (l *Layer) Open() (io.ReadSeekCloser, error) {
	if l.Digest == "" {
		return nil, errors.New("opening layer with empty digest")
	}

	blob, err := GetBlobsPath(l.Digest)
	if err != nil {
		return nil, err
	}

	return os.Open(blob)
}

func (l *Layer) Remove() error {
	if l.Digest == "" {
		return nil
	}

	// Ignore corrupt manifests to avoid blocking deletion of layers that are freshly orphaned
	ms, err := Manifests(true)
	if err != nil {
		return err
	}

	for _, m := range ms {
		for _, layer := range append(m.Layers, m.Config) {
			if layer.Digest == l.Digest {
				// something is using this layer
				return nil
			}
		}
	}

	blob, err := GetBlobsPath(l.Digest)
	if err != nil {
		return err
	}

	return os.Remove(blob)
}

type partKind int

const (
	kindHost partKind = iota
	kindNamespace
	kindModel
	kindTag
	kindDigest
)

func (k partKind) String() string {
	switch k {
	case kindHost:
		return "host"
	case kindNamespace:
		return "namespace"
	case kindModel:
		return "model"
	case kindTag:
		return "tag"
	case kindDigest:
		return "digest"
	default:
		return "unknown"
	}
}

// Name is a structured representation of a model name string, as defined by
// [ParseNameNoDefaults].
//
// It is not guaranteed to be valid. Use [Name.IsValid] to check if the name
// is valid.
type Name struct {
	Host      string
	Namespace string
	Model     string
	Tag       string
}

// ParseName parses and assembles a Name from a name string. The
// format of a valid name string is:
//
//	  s:
//		  { host } "/" { namespace } "/" { model } ":" { tag } "@" { digest }
//		  { host } "/" { namespace } "/" { model } ":" { tag }
//		  { host } "/" { namespace } "/" { model } "@" { digest }
//		  { host } "/" { namespace } "/" { model }
//		  { namespace } "/" { model } ":" { tag } "@" { digest }
//		  { namespace } "/" { model } ":" { tag }
//		  { namespace } "/" { model } "@" { digest }
//		  { namespace } "/" { model }
//		  { model } ":" { tag } "@" { digest }
//		  { model } ":" { tag }
//		  { model } "@" { digest }
//		  { model }
//		  "@" { digest }
//	  host:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" | "." | ":" }*
//	      length:  [1, 350]
//	  namespace:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" }*
//	      length:  [1, 80]
//	  model:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" | "." }*
//	      length:  [1, 80]
//	  tag:
//	      pattern: { alphanum | "_" } { alphanum | "-" | "_" | "." }*
//	      length:  [1, 80]
//	  digest:
//	      pattern: { alphanum | "_" } { alphanum | "-" | ":" }*
//	      length:  [1, 80]
//
// Most users should use [ParseName] instead, unless need to support
// different defaults than DefaultName.
//
// The name returned is not guaranteed to be valid. If it is not valid, the
// field values are left in an undefined state. Use [Name.IsValid] to check
// if the name is valid.
func ParseName(s string) Name {
	return Merge(ParseNameBare(s), DefaultName())
}

// ParseNameBare parses s as a name string and returns a Name. No merge with
// [DefaultName] is performed.
func ParseNameBare(s string) Name {
	var n Name
	var promised bool

	// "/" is an illegal tag character, so we can use it to split the host
	if strings.LastIndex(s, ":") > strings.LastIndex(s, "/") {
		s, n.Tag, _ = cutPromised(s, ":")
	}

	s, n.Model, promised = cutPromised(s, "/")
	if !promised {
		n.Model = s
		return n
	}

	s, n.Namespace, promised = cutPromised(s, "/")
	if !promised {
		n.Namespace = s
		return n
	}

	scheme, host, ok := strings.Cut(s, "://")
	if !ok {
		host = scheme
	}
	n.Host = host

	return n
}

// ParseNameFromFilepath parses a 4-part filepath as a Name. The parts are
// expected to be in the form:
//
// { host } "/" { namespace } "/" { model } "/" { tag }
func ParseNameFromFilepath(s string) (n Name) {
	parts := strings.Split(s, string(filepath.Separator))
	if len(parts) != 4 {
		return Name{}
	}

	n.Host = parts[0]
	n.Namespace = parts[1]
	n.Model = parts[2]
	n.Tag = parts[3]
	if !n.IsFullyQualified() {
		return Name{}
	}

	return n
}

// Merge merges the host, namespace, and tag parts of the two names,
// preferring the non-empty parts of a.
func Merge(a, b Name) Name {
	a.Host = cmp.Or(a.Host, b.Host)
	a.Namespace = cmp.Or(a.Namespace, b.Namespace)
	a.Tag = cmp.Or(a.Tag, b.Tag)
	return a
}

// String returns the name string, in the format that [ParseNameNoDefaults]
// accepts as valid, if [Name.IsValid] reports true; otherwise the empty
// string is returned.
func (n Name) String() string {
	var b strings.Builder
	if n.Host != "" {
		b.WriteString(n.Host)
		b.WriteByte('/')
	}
	if n.Namespace != "" {
		b.WriteString(n.Namespace)
		b.WriteByte('/')
	}
	b.WriteString(n.Model)
	if n.Tag != "" {
		b.WriteByte(':')
		b.WriteString(n.Tag)
	}
	return b.String()
}

// DisplayShortest returns a short string version of the name.
func (n Name) DisplayShortest() string {
	var sb strings.Builder

	if !strings.EqualFold(n.Host, DefaultHost) {
		sb.WriteString(n.Host)
		sb.WriteByte('/')
		sb.WriteString(n.Namespace)
		sb.WriteByte('/')
	} else if !strings.EqualFold(n.Namespace, DefaultNamespace) {
		sb.WriteString(n.Namespace)
		sb.WriteByte('/')
	}

	// always include model and tag
	sb.WriteString(n.Model)
	sb.WriteString(":")
	sb.WriteString(n.Tag)
	return sb.String()
}

// IsValidNamespace reports whether the provided string is a valid
// namespace.
func IsValidNamespace(s string) bool {
	return isValidPart(kindNamespace, s)
}

// IsValid reports whether all parts of the name are present and valid. The
// digest is a special case, and is checked for validity only if present.
//
// Note: The digest check has been removed as is planned to be added back in
// at a later time.
func (n Name) IsValid() bool {
	return n.IsFullyQualified()
}

// IsFullyQualified returns true if all parts of the name are present and
// valid without the digest.
func (n Name) IsFullyQualified() bool {
	parts := []string{
		n.Host,
		n.Namespace,
		n.Model,
		n.Tag,
	}
	for i, part := range parts {
		if !isValidPart(partKind(i), part) {
			return false
		}
	}
	return true
}

// Filepath returns a canonical filepath that represents the name with each part from
// host to tag as a directory in the form:
//
//	{host}/{namespace}/{model}/{tag}
//
// It uses the system's filepath separator and ensures the path is clean.
//
// It panics if the name is not fully qualified. Use [Name.IsFullyQualified]
// to check if the name is fully qualified.
func (n Name) Filepath() string {
	if !n.IsFullyQualified() {
		panic("illegal attempt to get filepath of invalid name")
	}
	return filepath.Join(
		n.Host,
		n.Namespace,
		n.Model,
		n.Tag,
	)
}

// LogValue returns a slog.Value that represents the name as a string.
func (n Name) LogValue() slog.Value {
	return slog.StringValue(n.String())
}

func (n Name) EqualFold(o Name) bool {
	return strings.EqualFold(n.Host, o.Host) &&
		strings.EqualFold(n.Namespace, o.Namespace) &&
		strings.EqualFold(n.Model, o.Model) &&
		strings.EqualFold(n.Tag, o.Tag)
}

func isValidLen(kind partKind, s string) bool {
	switch kind {
	case kindHost:
		return len(s) >= 1 && len(s) <= 350
	case kindTag:
		return len(s) >= 1 && len(s) <= 80
	default:
		return len(s) >= 1 && len(s) <= 80
	}
}

func isValidPart(kind partKind, s string) bool {
	if !isValidLen(kind, s) {
		return false
	}
	for i := range s {
		if i == 0 {
			if !isAlphanumericOrUnderscore(s[i]) {
				return false
			}
			continue
		}
		switch s[i] {
		case '_', '-':
		case '.':
			if kind == kindNamespace {
				return false
			}
		case ':':
			if kind != kindHost && kind != kindDigest {
				return false
			}
		default:
			if !isAlphanumericOrUnderscore(s[i]) {
				return false
			}
		}
	}
	return true
}

func isAlphanumericOrUnderscore(c byte) bool {
	return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_'
}

func cutLast(s, sep string) (before, after string, ok bool) {
	i := strings.LastIndex(s, sep)
	if i >= 0 {
		return s[:i], s[i+len(sep):], true
	}
	return s, "", false
}

// cutPromised cuts the last part of s at the last occurrence of sep. If sep is
// found, the part before and after sep are returned as-is unless empty, in
// which case they are returned as MissingPart, which will cause
// [Name.IsValid] to return false.
func cutPromised(s, sep string) (before, after string, ok bool) {
	before, after, ok = cutLast(s, sep)
	if !ok {
		return before, after, false
	}
	return cmp.Or(before, MissingPart), cmp.Or(after, MissingPart), true
}

type ModelPath struct {
	ProtocolScheme string
	Host           string
	Namespace      string
	Model          string
	Tag            string
}

const (
	DefaultHost           = "registry.ollama.ai"
	DefaultNamespace      = "library"
	DefaultTag            = "Q8_0"
	DefaultProtocolScheme = "https"
)

var (
	ErrInvalidImageFormat  = errors.New("invalid image format")
	ErrInvalidProtocol     = errors.New("invalid protocol scheme")
	ErrInsecureProtocol    = errors.New("insecure protocol http")
	ErrInvalidDigestFormat = errors.New("invalid digest format")
	ErrInvalidModelPath    = errors.New("invalid model path")
	ErrUnauthorized        = errors.New("unauthorized: access denied")
	ErrUnqualifiedName     = errors.New("unqualified name")
	ErrMaxRetriesExceeded  = errors.New("max retries exceeded")
	ErrPartStalled         = errors.New("part stalled")
	ErrDigestMismatch      = errors.New("digest mismatch, file must be downloaded again")
)

const (
	maxRetries                = 6
	numDownloadParts          = 10
	minDownloadPartSize int64 = 100 * MegaByte
	maxDownloadPartSize int64 = 1000 * MegaByte
)

func ParseModelPath(name string) ModelPath {
	mp := ModelPath{
		ProtocolScheme: DefaultProtocolScheme,
		Host:           DefaultHost,
		Namespace:      DefaultNamespace,
		Model:          "",
		Tag:            DefaultTag,
	}

	before, after, found := strings.Cut(name, "://")
	if found {
		mp.ProtocolScheme = before
		name = after
	}

	name = strings.ReplaceAll(name, string(os.PathSeparator), "/")
	parts := strings.Split(name, "/")
	switch len(parts) {
	case 3:
		mp.Host = parts[0]
		mp.Namespace = parts[1]
		mp.Model = parts[2]
	case 2:
		mp.Namespace = parts[0]
		mp.Model = parts[1]
	case 1:
		mp.Model = parts[0]
	}

	if m, tag, found := strings.Cut(mp.Model, ":"); found {
		mp.Model = m
		mp.Tag = strings.ToUpper(tag) // Make tags case-insensitive
	}

	return mp
}

func (mp ModelPath) GetNamespaceModel() string {
	return fmt.Sprintf("%s/%s", mp.Namespace, mp.Model)
}

func (mp ModelPath) GetFullTagname() string {
	return fmt.Sprintf("%s/%s/%s:%s", mp.Host, mp.Namespace, mp.Model, mp.Tag)
}

func (mp ModelPath) GetModelPath() string {
	return fmt.Sprintf("%s/%s", ModelsPath(), mp.Model)
}

func (mp ModelPath) GetModelTagPath() string {
	return fmt.Sprintf("%s/%s/%s", ModelsPath(), mp.Model, mp.Tag)
}

func (mp ModelPath) GetShortTagname() string {
	if mp.Host == DefaultHost {
		if mp.Namespace == DefaultNamespace {
			return fmt.Sprintf("%s:%s", mp.Model, mp.Tag)
		}
		return fmt.Sprintf("%s/%s:%s", mp.Namespace, mp.Model, mp.Tag)
	}
	return fmt.Sprintf("%s/%s/%s:%s", mp.Host, mp.Namespace, mp.Model, mp.Tag)
}

// GetManifestPath returns the path to the manifest file for the given model path, it is up to the caller to create the directory if it does not exist.
func (mp ModelPath) GetManifestPath() (string, error) {
	name := Name{
		Host:      mp.Host,
		Namespace: mp.Namespace,
		Model:     mp.Model,
		Tag:       mp.Tag,
	}
	if !name.IsValid() {
		return "", fs.ErrNotExist
	}
	return filepath.Join(ModelsPath(), "manifests", name.Filepath()), nil
}

func (mp ModelPath) BaseURL() *url.URL {
	return &url.URL{
		Scheme: mp.ProtocolScheme,
		Host:   mp.Host,
	}
}

func GetManifestPath() (string, error) {
	path := filepath.Join(ModelsPath(), "manifests")
	if err := os.MkdirAll(path, 0o755); err != nil {
		return "", err
	}

	return path, nil
}

func GetBlobsPath(digest string) (string, error) {
	// only accept actual sha256 digests
	pattern := "^sha256[:-][0-9a-fA-F]{64}$"
	re := regexp.MustCompile(pattern)

	if digest != "" && !re.MatchString(digest) {
		return "", ErrInvalidDigestFormat
	}

	digest = strings.ReplaceAll(digest, ":", "-")
	path := filepath.Join(ModelsPath(), "blobs", digest)
	dirPath := filepath.Dir(path)
	if digest == "" {
		dirPath = path
	}

	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return "", err
	}

	return path, nil
}

type Manifest struct {
	SchemaVersion int     `json:"schemaVersion"`
	MediaType     string  `json:"mediaType"`
	Config        Layer   `json:"config"`
	Layers        []Layer `json:"layers"`

	filepath string
	fi       os.FileInfo
	digest   string
}

func (m *Manifest) GetModelPath() string {
	return filepath.Join(ModelsPath(), m.Config.Digest)
}

func (m *Manifest) Size() (size int64) {
	for _, layer := range append(m.Layers, m.Config) {
		size += layer.Size
	}

	return
}

func (m *Manifest) Remove() error {
	if err := os.Remove(m.filepath); err != nil {
		return err
	}

	manifests, err := GetManifestPath()
	if err != nil {
		return err
	}

	return PruneDirectory(manifests)
}

func (m *Manifest) RemoveLayers() error {
	for _, layer := range append(m.Layers, m.Config) {
		if layer.Digest != "" {
			if err := layer.Remove(); errors.Is(err, os.ErrNotExist) {
				slog.Debug("layer does not exist", "digest", layer.Digest)
			} else if err != nil {
				return err
			}
		}
	}

	return nil
}

func ParseNamedManifest(n Name) (*Manifest, error) {
	if !n.IsFullyQualified() {
		return nil, fmt.Errorf("%w: %s", ErrUnqualifiedName, n)
	}

	manifests, err := GetManifestPath()
	if err != nil {
		return nil, err
	}

	p := filepath.Join(manifests, n.Filepath())

	var m Manifest
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}

	sha256sum := sha256.New()
	if err := json.NewDecoder(io.TeeReader(f, sha256sum)).Decode(&m); err != nil {
		return nil, err
	}

	m.filepath = p
	m.fi = fi
	m.digest = hex.EncodeToString(sha256sum.Sum(nil))

	return &m, nil
}

func Manifests(continueOnError bool) (map[Name]*Manifest, error) {
	manifests, err := GetManifestPath()
	if err != nil {
		return nil, err
	}

	// TODO(mxyng): use something less brittle
	matches, err := filepath.Glob(filepath.Join(manifests, "*", "*", "*", "*"))
	if err != nil {
		return nil, err
	}

	ms := make(map[Name]*Manifest)
	for _, match := range matches {
		fi, err := os.Stat(match)
		if err != nil {
			return nil, err
		}

		if !fi.IsDir() {
			rel, err := filepath.Rel(manifests, match)
			if err != nil {
				if !continueOnError {
					return nil, fmt.Errorf("%s %w", match, err)
				}
				slog.Warn("bad filepath", "path", match, "error", err)
				continue
			}

			n := ParseNameFromFilepath(rel)
			if !n.IsValid() {
				if !continueOnError {
					return nil, fmt.Errorf("%s %w", rel, err)
				}
				slog.Warn("bad manifest name", "path", rel)
				continue
			}

			m, err := ParseNamedManifest(n)
			if err != nil {
				if !continueOnError {
					return nil, fmt.Errorf("%s %w", n, err)
				}
				slog.Warn("bad manifest", "name", n, "error", err)
				continue
			}

			ms[n] = m
		}
	}

	return ms, nil
}

func makeRequest(ctx context.Context, method string, requestURL *url.URL, headers http.Header, body io.Reader, regOpts *registryOptions) (*http.Response, error) {
	if requestURL.Scheme != "http" && regOpts != nil && regOpts.Insecure {
		requestURL.Scheme = "http"
	}

	req, err := http.NewRequestWithContext(ctx, method, requestURL.String(), body)
	if err != nil {
		return nil, err
	}

	if headers != nil {
		req.Header = headers
	}

	if regOpts != nil {
		if regOpts.Token != "" {
			req.Header.Set("Authorization", "Bearer "+regOpts.Token)
		} else if regOpts.Username != "" && regOpts.Password != "" {
			req.SetBasicAuth(regOpts.Username, regOpts.Password)
		}
	}

	// We have to set as ollama otherwise the registry will hang up on us
	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s) Go/%s", Version, runtime.GOARCH, runtime.GOOS, runtime.Version()))

	if s := req.Header.Get("Content-Length"); s != "" {
		contentLength, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, err
		}

		req.ContentLength = contentLength
	}

	c := &http.Client{
		CheckRedirect: regOpts.CheckRedirect,
	}
	return c.Do(req)
}

func makeRequestWithRetry(ctx context.Context, method string, requestURL *url.URL, headers http.Header, body io.ReadSeeker, regOpts *registryOptions) (*http.Response, error) {
	for range 2 {
		resp, err := makeRequest(ctx, method, requestURL, headers, body, regOpts)
		if err != nil {
			if !errors.Is(err, context.Canceled) {
				slog.Info(fmt.Sprintf("request failed: %v", err))
			}

			return nil, err
		}

		switch {
		case resp.StatusCode == http.StatusUnauthorized:
			panic("unauthorized")
			// resp.Body.Close()

			// // Handle authentication error with one retry
			// challenge := parseRegistryChallenge(resp.Header.Get("www-authenticate"))
			// token, err := getAuthorizationToken(ctx, challenge)
			// if err != nil {
			// 	return nil, err
			// }
			// regOpts.Token = token
			// if body != nil {
			// 	_, err = body.Seek(0, io.SeekStart)
			// 	if err != nil {
			// 		return nil, err
			// 	}
			// }
		case resp.StatusCode == http.StatusNotFound:
			resp.Body.Close()
			return nil, os.ErrNotExist
		case resp.StatusCode >= http.StatusBadRequest:
			defer resp.Body.Close()
			responseBody, err := io.ReadAll(resp.Body)
			if err != nil {
				return nil, fmt.Errorf("%d: %s", resp.StatusCode, err)
			}
			return nil, fmt.Errorf("%d: %s", resp.StatusCode, responseBody)
		default:
			return resp, nil
		}
	}

	return nil, ErrUnauthorized
}

func pullModelManifest(ctx context.Context, mp ModelPath, regOpts *registryOptions) (*Manifest, error) {
	requestURL := mp.BaseURL().JoinPath("v2", mp.GetNamespaceModel(), "manifests", mp.Tag)

	headers := make(http.Header)
	headers.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, headers, nil, regOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var m Manifest
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}

	return &m, err
}

func GetManifest(mp ModelPath) (*Manifest, string, error) {
	fp, err := mp.GetManifestPath()
	if err != nil {
		return nil, "", err
	}

	f, err := os.Open(fp)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()

	sha256sum := sha256.New()

	var manifest Manifest
	if err := json.NewDecoder(io.TeeReader(f, sha256sum)).Decode(&manifest); err != nil {
		return nil, "", err
	}

	return &manifest, hex.EncodeToString(sha256sum.Sum(nil)), nil
}

func PruneDirectory(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}

	if info.IsDir() && info.Mode()&os.ModeSymlink == 0 {
		entries, err := os.ReadDir(path)
		if err != nil {
			return err
		}

		for _, entry := range entries {
			if err := PruneDirectory(filepath.Join(path, entry.Name())); err != nil {
				return err
			}
		}

		entries, err = os.ReadDir(path)
		if err != nil {
			return err
		}

		if len(entries) > 0 {
			return nil
		}

		return os.Remove(path)
	}

	return nil
}

func (p *blobDownloadPart) Name() string {
	return strings.Join([]string{
		p.blobDownload.Name, "partial", strconv.Itoa(p.N),
	}, "-")
}

func (p *blobDownloadPart) StartsAt() int64 {
	return p.Offset + p.Completed.Load()
}

func (p *blobDownloadPart) StopsAt() int64 {
	return p.Offset + p.Size
}

func (p *blobDownloadPart) Write(b []byte) (n int, err error) {
	n = len(b)
	p.blobDownload.Completed.Add(int64(n))
	p.lastUpdatedMu.Lock()
	p.lastUpdated = time.Now()
	p.lastUpdatedMu.Unlock()
	return n, nil
}

func (b *blobDownload) newPart(offset, size int64) error {
	part := blobDownloadPart{blobDownload: b, Offset: offset, Size: size, N: len(b.Parts)}
	if err := b.writePart(part.Name(), &part); err != nil {
		return err
	}

	b.Parts = append(b.Parts, &part)
	return nil
}

func (b *blobDownload) readPart(partName string) (*blobDownloadPart, error) {
	var part blobDownloadPart
	partFile, err := os.Open(partName)
	if err != nil {
		return nil, err
	}
	defer partFile.Close()

	if err := json.NewDecoder(partFile).Decode(&part); err != nil {
		return nil, err
	}

	part.blobDownload = b
	return &part, nil
}

func (b *blobDownload) writePart(partName string, part *blobDownloadPart) error {
	partFile, err := os.OpenFile(partName, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer partFile.Close()

	return json.NewEncoder(partFile).Encode(part)
}

func (b *blobDownload) acquire() {
	b.references.Add(1)
}

func (b *blobDownload) release() {
	if b.references.Add(-1) == 0 {
		b.CancelFunc()
	}
}

func (b *blobDownload) Wait(ctx context.Context) error {
	b.acquire()
	defer b.release()

	var lastcompleted int64
	ticker := time.NewTicker(time.Second)
	for {
		select {
		case <-b.done:
			pbar.SetProgress(b.Completed.Load())
			return b.err
		case <-ticker.C:
			speed := (b.Completed.Load() - lastcompleted) // bytes per second
			var estimate time.Duration
			if speed > 0 {
				estimate = time.Duration((b.Total-b.Completed.Load())/speed) * time.Second
			}
			pbar.PostfixText = fmt.Sprintf("%8s/s %s", HumanBytes(speed), estimate)
			pbar.SetProgress(b.Completed.Load())
			lastcompleted = b.Completed.Load()
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

type downloadOpts struct {
	mp      ModelPath
	digest  string
	regOpts *registryOptions
}

var blobDownloadManager sync.Map

type blobDownload struct {
	Name   string
	Digest string

	Total     int64
	Completed atomic.Int64

	Parts []*blobDownloadPart

	context.CancelFunc

	done       chan struct{}
	err        error
	references atomic.Int32
}

type blobDownloadPart struct {
	N         int
	Offset    int64
	Size      int64
	Completed atomic.Int64

	lastUpdatedMu sync.Mutex
	lastUpdated   time.Time

	*blobDownload `json:"-"`
}

// downloadBlob downloads a blob from the registry and stores it in the blobs directory
func downloadBlob(ctx context.Context, opts downloadOpts) (bool, error) {
	fp, err := GetBlobsPath(opts.digest)
	if err != nil {
		return false, err
	}

	_, err = os.Stat(fp)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		return false, err
	default:
		fmt.Printf("%s cached already\n", opts.digest[7:15])
		return true, nil
	}

	data, ok := blobDownloadManager.LoadOrStore(opts.digest, &blobDownload{Name: fp, Digest: opts.digest})
	download := data.(*blobDownload)
	if !ok {
		requestURL := opts.mp.BaseURL()
		requestURL = requestURL.JoinPath("v2", opts.mp.GetNamespaceModel(), "blobs", opts.digest)
		if err := download.Prepare(ctx, requestURL, opts.regOpts); err != nil {
			blobDownloadManager.Delete(opts.digest)
			return false, err
		}
		pbar.Reset()
		pbar.Max = download.Total
		pbar.PrefixText = fmt.Sprintf("Pulling %s... ", opts.digest[7:15])
		//nolint:contextcheck
		go download.Run(context.Background(), requestURL, opts.regOpts)
	}

	return false, download.Wait(ctx)
}

func (b *blobDownload) Prepare(ctx context.Context, requestURL *url.URL, opts *registryOptions) error {
	partFilePaths, err := filepath.Glob(b.Name + "-partial-*")
	if err != nil {
		return err
	}

	b.done = make(chan struct{})

	for _, partFilePath := range partFilePaths {
		part, err := b.readPart(partFilePath)
		if err != nil {
			return err
		}

		b.Total += part.Size
		b.Completed.Add(part.Completed.Load())
		b.Parts = append(b.Parts, part)
	}

	if len(b.Parts) == 0 {
		resp, err := makeRequestWithRetry(ctx, http.MethodHead, requestURL, nil, nil, opts)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		b.Total, _ = strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)

		size := b.Total / numDownloadParts
		switch {
		case size < minDownloadPartSize:
			size = minDownloadPartSize
		case size > maxDownloadPartSize:
			size = maxDownloadPartSize
		}

		var offset int64
		for offset < b.Total {
			if offset+size > b.Total {
				size = b.Total - offset
			}

			if err := b.newPart(offset, size); err != nil {
				return err
			}

			offset += size
		}
	}

	// slog.Info(fmt.Sprintf("downloading %s in %d %s part(s)", b.Digest[7:15], len(b.Parts), HumanBytes(b.Parts[0].Size)))
	return nil
}

func (b *blobDownload) Run(ctx context.Context, requestURL *url.URL, opts *registryOptions) {
	defer close(b.done)
	b.err = b.run(ctx, requestURL, opts)
}

func (b *blobDownload) run(ctx context.Context, requestURL *url.URL, opts *registryOptions) error {
	defer blobDownloadManager.Delete(b.Digest)
	ctx, b.CancelFunc = context.WithCancel(ctx)

	file, err := os.OpenFile(b.Name+"-partial", os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()
	// setSparse(file)

	_ = file.Truncate(b.Total)

	directURL, err := func() (*url.URL, error) {
		ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
		defer cancel()

		backoff := newBackoff(10 * time.Second)
		for {
			// shallow clone opts to be used in the closure
			// without affecting the outer opts.
			newOpts := new(registryOptions)
			*newOpts = *opts

			newOpts.CheckRedirect = func(req *http.Request, via []*http.Request) error {
				if len(via) > 10 {
					return errors.New("maximum redirects exceeded (10) for directURL")
				}

				// if the hostname is the same, allow the redirect
				if req.URL.Hostname() == requestURL.Hostname() {
					return nil
				}

				// stop at the first redirect that is not
				// the same hostname as the original
				// request.
				return http.ErrUseLastResponse
			}

			resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, nil, nil, newOpts)
			if err != nil {
				slog.Warn("failed to get direct URL; backing off and retrying", "err", err)
				if err := backoff(ctx); err != nil {
					return nil, err
				}
				continue
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusTemporaryRedirect && resp.StatusCode != http.StatusOK {
				return nil, fmt.Errorf("unexpected status code %d", resp.StatusCode)
			}
			return resp.Location()
		}
	}()
	if err != nil {
		return err
	}

	g, inner := errgroup.WithContext(ctx)
	g.SetLimit(numDownloadParts)
	for i := range b.Parts {
		part := b.Parts[i]
		if part.Completed.Load() == part.Size {
			continue
		}

		g.Go(func() error {
			var err error
			for try := 0; try < maxRetries; try++ {
				w := io.NewOffsetWriter(file, part.StartsAt())
				err = b.downloadChunk(inner, directURL, w, part)
				switch {
				case errors.Is(err, context.Canceled), errors.Is(err, syscall.ENOSPC):
					// return immediately if the context is canceled or the device is out of space
					return err
				case errors.Is(err, ErrPartStalled):
					try--
					continue
				case err != nil:
					sleep := time.Second * time.Duration(math.Pow(2, float64(try)))
					slog.Info(fmt.Sprintf("%s part %d attempt %d failed: %v, retrying in %s", b.Digest[7:19], part.N, try, err, sleep))
					time.Sleep(sleep)
					continue
				default:
					return nil
				}
			}

			return fmt.Errorf("%w: %w", ErrMaxRetriesExceeded, err)
		})
	}

	if err := g.Wait(); err != nil {
		return err
	}

	// explicitly close the file so we can rename it
	if err := file.Close(); err != nil {
		return err
	}

	for i := range b.Parts {
		if err := os.Remove(file.Name() + "-" + strconv.Itoa(i)); err != nil {
			return err
		}
	}

	if err := os.Rename(file.Name(), b.Name); err != nil {
		return err
	}

	return nil
}

func newBackoff(maxBackoff time.Duration) func(ctx context.Context) error {
	var n int
	return func(ctx context.Context) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		n++

		// n^2 backoff timer is a little smoother than the
		// common choice of 2^n.
		d := min(time.Duration(n*n)*10*time.Millisecond, maxBackoff)
		// Randomize the delay between 0.5-1.5 x msec, in order
		// to prevent accidental "thundering herd" problems.
		d = time.Duration(float64(d) * (rand.Float64() + 0.5))
		t := time.NewTimer(d)
		defer t.Stop()
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
			return nil
		}
	}
}

func (b *blobDownload) downloadChunk(ctx context.Context, requestURL *url.URL, w io.Writer, part *blobDownloadPart) error {
	g, ctx := errgroup.WithContext(ctx)
	g.Go(func() error {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL.String(), nil)
		if err != nil {
			return err
		}
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", part.StartsAt(), part.StopsAt()-1))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		n, err := io.CopyN(w, io.TeeReader(resp.Body, part), part.Size-part.Completed.Load())
		if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, io.ErrUnexpectedEOF) {
			// rollback progress
			b.Completed.Add(-n)
			return err
		}

		part.Completed.Add(n)
		if err := b.writePart(part.Name(), part); err != nil {
			return err
		}

		// return nil or context.Canceled or UnexpectedEOF (resumable)
		return err
	})

	g.Go(func() error {
		ticker := time.NewTicker(time.Second)
		for {
			select {
			case <-ticker.C:
				if part.Completed.Load() >= part.Size {
					return nil
				}

				part.lastUpdatedMu.Lock()
				lastUpdated := part.lastUpdated
				part.lastUpdatedMu.Unlock()

				if !lastUpdated.IsZero() && time.Since(lastUpdated) > 5*time.Second {
					const msg = "%s part %d stalled; retrying. If this persists, press ctrl-c to exit, then try to find a faster host."
					slog.Info(fmt.Sprintf(msg, b.Digest[7:15], part.N))
					// reset last updated
					part.lastUpdatedMu.Lock()
					part.lastUpdated = time.Time{}
					part.lastUpdatedMu.Unlock()
					return ErrPartStalled
				}
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	})

	return g.Wait()
}

func verifyBlob(digest string) error {
	fp, err := GetBlobsPath(digest)
	if err != nil {
		return err
	}

	f, err := os.Open(fp)
	if err != nil {
		return err
	}
	defer f.Close()

	fileDigest, _ := GetSHA256Digest(f)
	if digest != fileDigest {
		return fmt.Errorf("%w: want %s, got %s", ErrDigestMismatch, digest, fileDigest)
	}

	return nil
}

// GetSHA256Digest returns the SHA256 hash of a given buffer and returns it, and the size of buffer
func GetSHA256Digest(r io.Reader) (string, int64) {
	h := sha256.New()
	n, err := io.Copy(h, r)
	if err != nil {
		panic(err)
	}

	return fmt.Sprintf("sha256:%x", h.Sum(nil)), n
}

func deleteUnusedLayers(deleteMap Set) error {
	// Ignore corrupt manifests to avoid blocking deletion of layers that are freshly orphaned
	manifests, err := Manifests(true)
	if err != nil {
		return err
	}

	for _, manifest := range manifests {
		for _, layer := range manifest.Layers {
			delete(deleteMap, layer.Digest)
		}

		delete(deleteMap, manifest.Config.Digest)
	}

	// only delete the files which are still in the deleteMap
	for k := range deleteMap {
		fp, err := GetBlobsPath(k)
		if err != nil {
			slog.Info(fmt.Sprintf("couldn't get file path for '%s': %v", k, err))
			continue
		}
		if err := os.Remove(fp); err != nil {
			slog.Info(fmt.Sprintf("couldn't remove file '%s': %v", fp, err))
			continue
		}
	}

	return nil
}

func BuildPruneMap(mp ModelPath) (Set, error) {
	prune := make(Set)
	manifest, _, err := GetManifest(mp)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		slog.Warn("pulling model with bad existing manifest", "name", mp.GetFullTagname(), "error", err)
	default: // exists
		for _, l := range manifest.Layers {
			prune.Add(l.Digest)
		}
		if manifest.Config.Digest != "" {
			prune.Add(manifest.Config.Digest)
		}
	}
	return prune, nil
}

func PullManifest(ctx context.Context, mp ModelPath, regOpts *registryOptions) (*Manifest, error) {
	fmt.Print("Pulling manifest...")
	manifest, err := pullModelManifest(ctx, mp, regOpts)
	if err != nil {
		return nil, fmt.Errorf("pull model manifest: %s", err)
	}
	fmt.Println("Done")
	return manifest, nil
}

func PullLayers(ctx context.Context, mp ModelPath, regOpts *registryOptions, manifest *Manifest, prune Set) error {
	var layers []Layer
	layers = append(layers, manifest.Layers...)
	if manifest.Config.Digest != "" {
		layers = append(layers, manifest.Config)
	}
	skipVerify := make(map[string]bool)
	for _, layer := range layers {
		cacheHit, err := downloadBlob(ctx, downloadOpts{
			mp:      mp,
			digest:  layer.Digest,
			regOpts: regOpts,
		})
		if err != nil {
			return err
		}
		skipVerify[layer.Digest] = cacheHit
		prune.Remove(layer.Digest)
	}
	prune.Remove(manifest.Config.Digest)
	if err := VerifyLayers(layers, skipVerify); err != nil {
		return err
	}
	return nil
}

func WriteManifest(mp ModelPath, manifest *Manifest) error {
	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	fp, err := mp.GetManifestPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
		return err
	}

	err = os.WriteFile(fp, manifestJSON, 0o644)
	if err != nil {
		slog.Info(fmt.Sprintf("couldn't write to %s", fp))
		return err
	}
	return nil
}

func PruneLayers(prune Set) error {
	if len(prune) > 0 {
		fmt.Print("Pruning unused layers...")
		if err := deleteUnusedLayers(prune); err != nil {
			fmt.Fprintf(os.Stderr, "couldn't remove unused layers: %v", err)
		}
		fmt.Println("Done")
	}
	return nil
}

func VerifyLayers(layers []Layer, skipVerify map[string]bool) error {
	pbar.Reset()
	pbar.Max = int64(len(layers))
	pbar.ShowPercentage = true
	for _, layer := range layers {
		pbar.PrefixText = fmt.Sprintf("Verifying %s... ", layer.Digest[7:15])
		pbar.Tick()
		if skipVerify[layer.Digest] {
			continue
		}
		if err := verifyBlob(layer.Digest); err != nil {
			if errors.Is(err, ErrDigestMismatch) {
				fp, err := GetBlobsPath(layer.Digest)
				if err != nil {
					return err
				}
				if err := os.Remove(fp); err != nil {
					slog.Info(fmt.Sprintf("couldn't remove file with digest mismatch '%s': %v", fp, err))
				}
			}
			return err
		}
	}
	return nil
}

func LinkModel(mp ModelPath, manifest *Manifest) error {
	if err := os.MkdirAll(mp.GetModelPath(), 0o755); err != nil {
		return err
	}

	for _, layer := range manifest.Layers {
		old, err := GetBlobsPath(layer.Digest)
		if err != nil {
			return err
		}

		if layer.MediaType == "application/vnd.ollama.image.model" {
			if err := os.Symlink(old, mp.GetModelTagPath()); err != nil && !os.IsExist(err) {
				return err
			}
		}
	}
	return nil
}
