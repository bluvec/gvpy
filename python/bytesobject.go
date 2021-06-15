package python

// #include "py.h"
import "C"
import "unsafe"

// int PyBytes_Check(PyObject *o)
func PyBytes_Check(o *PyObject) bool {
	return int(C.__PyBytes_Check(go2c(o))) == 1
}

// PyObject* PyBytes_FromString(const char *v)
// Return new reference.
func PyBytes_FromString(v string) *PyObject {
	cv := C.CString(v)
	defer C.free(unsafe.Pointer(cv))

	return c2go(C.__PyBytes_FromString(cv))
}

// PyObject* PyBytes_FromStringAndSize(const char *v, Py_ssize_t len)
// Return: new reference.
// Ref: https://docs.python.org/3.8/c-api/bytes.html#c.PyBytes_FromStringAndSize
func PyBytes_FromBytes(v []byte) *PyObject {
	cv := C.CBytes(v)
	defer C.free(cv)

	return c2go(C.__PyBytes_FromStringAndSize((*C.char)(cv), C.Py_ssize_t(len(v))))
}

// Py_ssize_t PyBytes_Size(PyObject *o)
func PyBytes_Size(o *PyObject) int {
	return int(C.__PyBytes_Size(go2c(o)))
}

// char* PyBytes_AsString(PyObject *o)
// Ref: https://docs.python.org/3.8/c-api/bytes.html#c.PyBytes_AsString
func PyBytes_AsString(o *PyObject) string {
	return C.GoStringN(C.__PyBytes_AsString(go2c(o)), C.int(C.__PyBytes_Size(go2c(o))))
}

// char* PyBytes_AsString(PyObject *o)
// Ref: https://docs.python.org/3.8/c-api/bytes.html#c.PyBytes_AsString
func PyBytes_AsBytes(o *PyObject) []byte {
	return C.GoBytes(unsafe.Pointer(C.__PyBytes_AsString(go2c(o))), C.int(C.__PyBytes_Size(go2c(o))))
}
