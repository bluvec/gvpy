package python

// #include "py.h"
import "C"
import (
	"fmt"
	"unsafe"
)

// Ref: https://docs.python.org/3.8/c-api/sys.html?#c.PySys_GetObject
// Return: borrowed reference.
func PySys_GetObject(name string) *PyObject {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	return c2go(C.__PySys_GetObject(cname))
}

// Ref: https://docs.python.org/3.8/c-api/sys.html?#c.PySys_SetObject
func PySys_SetObject(name string, o *PyObject) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	ret := int(C.__PySys_SetObject(cname, go2c(o)))
	if ret == 0 {
		return nil
	} else {
		return fmt.Errorf("error to PySys_SetObject")
	}
}

// void PySys_SetPath(const wchar_t *path)
// https://docs.python.org/3.8/c-api/sys.html?#c.PySys_SetPath
func PySys_SetPath(path string) {
	panic("not implemented")
}
