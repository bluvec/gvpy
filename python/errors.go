package python

// #include "py.h"
import "C"

// PyObject* PyErr_Occurred()
// Return: borrowed reference.
func PyErr_Occurred() *PyObject {
	return c2go(C.__PyErr_Occurred())
}

// void PyErr_PrintEx(int set_sys_last_vars)
func PyErr_PrintEx(set_sys_last_vars int) {
	C.__PyErr_PrintEx(C.int(set_sys_last_vars))
}

// void PyErr_Print()
func PyErr_Print() {
	C.__PyErr_Print()
}
