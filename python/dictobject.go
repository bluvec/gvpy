package python

// #include "py.h"
import "C"
import (
	"fmt"
	"unsafe"
)

// int PyDict_Check(PyObject *p)
func PyDict_Check(o *PyObject) bool {
	return int(C.__PyDict_Check(go2c(o))) == 1
}

// PyObject* PyDict_New()
// Return new reference.
func PyDict_New() *PyObject {
	return c2go(C.__PyDict_New())
}

// PyObject* PyDict_GetItem(PyObject *p, PyObject *key)
// Return borrowed reference.
func PyDict_GetItem(o, key *PyObject) *PyObject {
	return c2go(C.__PyDict_GetItem(go2c(o), go2c(key)))
}

// PyObject* PyDict_GetItemString(PyObject *p, const char *key)
// Return borrowed reference.
func PyDict_GetItemString(o *PyObject, key string) *PyObject {
	ckey := C.CString(key)
	defer C.free(unsafe.Pointer(ckey))

	return c2go(C.__PyDict_GetItemString(go2c(o), ckey))
}

// int PyDict_SetItem(PyObject *p, PyObject *key, PyObject *val)
func PyDict_SetItem(o, key, val *PyObject) error {
	if rc := int(C.__PyDict_SetItem(go2c(o), go2c(key), go2c(val))); rc != 0 {
		return fmt.Errorf("python: error to set item to dict")
	}
	return nil
}

// int PyDict_SetItemString(PyObject *p, const char *key, PyObject *val)
func PyDict_SetItemString(o *PyObject, key string, val *PyObject) error {
	ckey := C.CString(key)
	defer C.free(unsafe.Pointer(ckey))

	if rc := int(C.__PyDict_SetItemString(go2c(o), ckey, go2c(val))); rc != 0 {
		return fmt.Errorf("cpython: error to set item to dict, key: %v", key)
	}
	return nil
}

// int PyDict_DelItem(PyObject *p, PyObject *key)
func PyDict_DelItem(o, key *PyObject) error {
	if rc := int(C.__PyDict_DelItem(go2c(o), go2c(key))); rc != 0 {
		return fmt.Errorf("cpython: error to del item from dict")
	}
	return nil
}

// int PyDict_DelItemString(PyObject *p, const char *key)
func PyDict_DelItemString(o *PyObject, key string) error {
	ckey := C.CString(key)
	defer C.free(unsafe.Pointer(ckey))

	if rc := int(C.__PyDict_DelItemString(go2c(o), ckey)); rc != 0 {
		return fmt.Errorf("cpython: error to del item from dict, key: '%v'", key)
	}
	return nil
}

// int PyDict_Contains(PyObject *p, PyObject *key)
func PyDict_Contains(o, key *PyObject) bool {
	return int(C.__PyDict_Contains(go2c(o), go2c(key))) == 1
}

// void PyDict_Clear(PyObject *p)
func PyDict_Clear(o *PyObject) {
	C.__PyDict_Clear(go2c(o))
}

// PyObject* PyDict_Items(PyObject *p)
// Return new reference.
func PyDict_Items(o *PyObject) *PyObject {
	return c2go(C.__PyDict_Items(go2c(o)))
}

// PyObject* PyDict_Keys(PyObject *p)
// Return new reference.
func PyDict_Keys(o *PyObject) *PyObject {
	return c2go(C.__PyDict_Keys(go2c(o)))
}

// PyObject* PyDict_Values(PyObject *p)
// Return: new reference.
func PyDict_Values(o *PyObject) *PyObject {
	return c2go(C.__PyDict_Values(go2c(o)))
}

// Py_ssize_t PyDict_Size(PyObject *p)
func PyDict_Size(o *PyObject) int {
	return int(C.__PyDict_Size(go2c(o)))
}

// int PyDict_Next(PyObject *p, Py_ssize_t *ppos, PyObject **pkey, PyObject **pvalue)
func PyDict_Next(o *PyObject, ppos *int, pkey, pval **PyObject) bool {
	if ppos == nil {
		return false
	}

	cpos := C.Py_ssize_t(*ppos)
	ckey := go2c(*pkey)
	cval := go2c(*pval)

	rc := int(C.__PyDict_Next(go2c(o), &cpos, &ckey, &cval))

	*ppos = int(cpos)
	*pkey = c2go(ckey)
	*pval = c2go(cval)

	if rc == 0 {
		return false
	} else {
		return true
	}
}

// int PyDict_Merge(PyObject *a, PyObject *b, int override)
func PyDict_Merge(a, b *PyObject, override bool) error {
	var iov C.int
	if override {
		iov = C.int(1)
	} else {
		iov = C.int(0)
	}

	rc := int(C.__PyDict_Merge(go2c(a), go2c(b), iov))
	if rc != 0 {
		return fmt.Errorf("cpython: error to merge two PyDicts")
	}

	return nil
}

// int PyDict_Update(PyObject *a, PyObject *b)
func PyDict_Update(a, b *PyObject) error {
	rc := int(C.__PyDict_Update(go2c(a), go2c(b)))
	if rc != 0 {
		return fmt.Errorf("cpython: error to update PyDict")
	}
	return nil
}
