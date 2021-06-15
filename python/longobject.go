package python

// #include "py.h"
import "C"

// int PyLong_Check(PyObject *p)
func PyLong_Check(o *PyObject) bool {
	return int(C.__PyLong_Check(go2c(o))) == 1
}

// long PyLong_AsLong(PyObject *obj)
func PyLong_AsLong(o *PyObject) int {
	return int(C.__PyLong_AsLong(go2c(o)))
}

// PyObject* PyLong_FromLong(long v)
// Return: new reference.
func PyLong_FromLong(v int) *PyObject {
	return c2go(C.__PyLong_FromLong(C.long(v)))
}
