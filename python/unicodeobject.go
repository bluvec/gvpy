package python

// #include "py.h"
import "C"
import "unsafe"

// int PyUnicode_Check(PyObject *o)
// Ref: https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_Check
func PyUnicode_Check(o *PyObject) bool {
	return int(C.__PyUnicode_Check(go2c(o))) == 1
}

// PyObject *PyUnicode_FromString(const char *u)
// Return: new reference.
// Ref: https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_FromString
func PyUnicode_FromString(u string) *PyObject {
	cu := C.CString(u)
	defer C.free(unsafe.Pointer(cu))

	return c2go(C.__PyUnicode_FromString(cu))
}

// const char* PyUnicode_AsUTF8(PyObject *unicode)
// Ref: https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_AsUTF8
func PyUnicode_AsString(unicode *PyObject) string {
	cu := C.__PyUnicode_AsUTF8(go2c(unicode))
	return C.GoString(cu)
}
