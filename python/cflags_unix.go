// +build !windows
// +build !custom
// +build !py36
// +build !py37

package python

// #cgo pkg-config: python3-embed
// #include "py.h"
import "C"

const PyDelimiter = ":"
