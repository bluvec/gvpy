package python

// #include "py.h"
import "C"

// int PyModule_Check(PyObject *p)
// Ref: https://docs.python.org/3/c-api/module.html?#c.PyModule_Check
func PyModule_Check(o *PyObject) bool {
	return int(C.__PyModule_Check(go2c(o))) == 1
}
