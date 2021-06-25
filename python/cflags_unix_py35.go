// +build !windows
// +build !custom
// +build py35

package python

// #cgo pkg-config: python-3.5m
// #include "py.h"
import "C"

const PyDelimiter = ":"
