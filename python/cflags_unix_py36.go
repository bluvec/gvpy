// +build !windows
// +build !custom
// +build py36

package python

// #cgo pkg-config: python-3.6m
// #include "py.h"
import "C"

const PyDelimiter = ":"
