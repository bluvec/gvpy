package python

// #include "py.h"
import "C"

var Py_True = &PyObject{p: C.__Py_True()}
var Py_False = &PyObject{p: C.__Py_False()}

// int PyBool_Check(PyObject *o)
func PyBool_Check(o *PyObject) bool {
	return int(C.__PyBool_Check(go2c(o))) == 1
}

// PyObject* PyBool_FromLong(long v);
// Return new reference.
func PyBool_FromLong(v int) *PyObject {
	return c2go(C.__PyBool_FromLong(C.long(v)))
}

// Return new reference.
func PyBool_FromBool(v bool) *PyObject {
	if v {
		return PyBool_FromLong(1)
	} else {
		return PyBool_FromLong(0)
	}
}

func PyBool_IsTure(o *PyObject) bool {
	return go2c(o) == Py_True.p
}

func PyBool_IsFalse(o *PyObject) bool {
	return go2c(o) == Py_False.p
}
