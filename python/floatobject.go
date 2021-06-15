package python

// #include "py.h"
import "C"

// int PyFloat_Check(PyObject *p)
func PyFloat_Check(o *PyObject) bool {
	return int(C.__PyFloat_Check(go2c(o))) == 1
}

// double PyFloat_AsDouble(PyObject *pyfloat)
func PyFloat_AsDouble(o *PyObject) float64 {
	return float64(C.__PyFloat_AsDouble(go2c(o)))
}

// PyObject* PyFloat_FromDouble(double v)
// Return: new reference.
func PyFloat_FromDouble(v float64) *PyObject {
	return c2go(C.__PyFloat_FromDouble(C.double(v)))
}
