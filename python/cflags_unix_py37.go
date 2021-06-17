// +build !windows
// +build !custom
// +build py37

package python

// #cgo pkg-config: python-3.7m
// #include "py.h"
import "C"

const PyDelimiter = ":"
