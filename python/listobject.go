package python

// #include "py.h"
import "C"
import "fmt"

// int PyList_Check(PyObject *p)
func PyList_Check(o *PyObject) bool {
	return int(C.__PyList_Check(go2c(o))) == 1
}

// PyObject* PyList_New(Py_ssize_t len)
// Return: new reference.
func PyList_New(len int) *PyObject {
	return c2go(C.__PyList_New(C.Py_ssize_t(len)))
}

// Py_ssize_t PyList_Size(PyObject *list)
func PyList_Size(o *PyObject) int {
	return int(C.__PyList_Size(go2c(o)))
}

// PyObject* PyList_GetItem(PyObject *list, Py_ssize_t index)
// Return: borrowed reference.
func PyList_GetItem(o *PyObject, index int) *PyObject {
	return c2go(C.__PyList_GetItem(go2c(o), C.Py_ssize_t(index)))
}

// int PyList_SetItem(PyObject *list, Py_ssize_t index, PyObject *item)
func PyList_SetItem(o *PyObject, index int, item *PyObject) error {
	if rc := int(C.__PyList_SetItem(go2c(o), C.Py_ssize_t(index), go2c(item))); rc != 0 {
		return fmt.Errorf("cpython: error to set item of PyList")
	}
	return nil
}

// int PyList_Insert(PyObject *list, Py_ssize_t index, PyObject *item)
// Ref: https://docs.python.org/3.8/c-api/list.html?#c.PyList_Insert
func PyList_Insert(o *PyObject, index int, item *PyObject) error {
	if rc := int(C.__PyList_Insert(go2c(o), C.Py_ssize_t(index), go2c(item))); rc != 0 {
		return fmt.Errorf("cpython: error to insert item to PyList")
	}
	return nil
}

// int PyList_Append(PyObject *list, PyObject *item)
func PyList_Append(o *PyObject, item *PyObject) error {
	if rc := int(C.__PyList_Append(go2c(o), go2c(item))); rc != 0 {
		return fmt.Errorf("cpython: error to append item to PyList")
	}
	return nil
}

// PyObject* PyList_AsTuple(PyObject *list)
// Return: new reference.
func PyList_AsTuple(o *PyObject) *PyObject {
	return c2go(C.__PyList_AsTuple(go2c(o)))
}
