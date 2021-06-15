package python

// #include "py.h"
import "C"

// int PyFunction_Check(PyObject *o)
func PyFunction_Check(o *PyObject) bool {
	return int(C.__PyFunction_Check(go2c(o))) == 1
}
