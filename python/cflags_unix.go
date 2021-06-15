// +build !windows
// +build !custom

package python

// #cgo pkg-config: python-3.8
// #cgo CFLAGS: -g -O3 -fPIC -ffast-math
// #cgo CFLAGS: -I${SRCDIR}/3rd/include/python3.8
// #cgo LDFLAGS: -L${SRCDIR}/3rd/lib -Wl,-rpath,${SRCDIR}/3rd/lib -lpython3.8
// #include "py.h"
import "C"

const PyDelimiter = ":"
