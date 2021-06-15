package python

// #include "py.h"
import "C"
import "fmt"

// int PyTuple_Check(PyObject *p)
func PyTuple_Check(o *PyObject) bool {
	return int(C.__PyTuple_Check(go2c(o))) == 1
}

// PyObject* PyTuple_New(Py_ssize_t len)
// Return: new reference.
func PyTuple_New(len int) *PyObject {
	return c2go(C.__PyTuple_New(C.Py_ssize_t(len)))
}

// Py_ssize_t PyTuple_Size(PyObject *p)
func PyTuple_Size(o *PyObject) int {
	return int(C.__PyTuple_Size(go2c(o)))
}

// PyObject* PyTuple_GetItem(PyObject *p, Py_ssize_t pos)
// Return: borrowed reference.
func PyTuple_GetItem(o *PyObject, pos int) *PyObject {
	return c2go(C.__PyTuple_GetItem(go2c(o), C.Py_ssize_t(pos)))
}

// int PyTuple_SetItem(PyObject *p, Py_ssize_t pos, PyObject *o)
// Note: This function "steals" a reference to `item` and discards a reference to an item already in the tuple at the affected position.
func PyTuple_SetItem(o *PyObject, pos int, item *PyObject) error {
	if rc := int(C.__PyTuple_SetItem(go2c(o), C.Py_ssize_t(pos), go2c(item))); rc != 0 {
		return fmt.Errorf("cpython: error to set item to PyTuple, pos: %v", pos)
	}
	return nil
}
