// +build !windows
// +build !custom

package python

// #cgo pkg-config: python3-embed
// #include "py.h"
import "C"

const PyDelimiter = ":"
