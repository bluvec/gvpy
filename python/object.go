package python

// #include "py.h"
import "C"
import (
	"fmt"
	"unsafe"
)

type PyObject struct {
	p *C.PyObject
}

var Py_None = &PyObject{p: C.__Py_None()}

// Stringer of PyObject
func (o *PyObject) String() string {
	uo := o.Str()
	defer uo.Py_Clear()

	return PyUnicode_AsString(uo)
}

func go2c(o *PyObject) *C.PyObject {
	if o == nil {
		return nil
	} else {
		return o.p
	}
}

func c2go(co *C.PyObject) *PyObject {
	switch co {
	case nil:
		return nil
	case Py_None.p:
		return Py_None
	case Py_True.p:
		return Py_True
	case Py_False.p:
		return Py_False
	default:
		return &PyObject{p: co}
	}
}

func Py_XINCREF(o *PyObject) {
	if o != nil && o != Py_None && o != Py_True && o != Py_False {
		C.__Py_XINCREF(go2c(o))
	}
}

func Py_XDECREF(o *PyObject) {
	if o != nil && o != Py_None && o != Py_True && o != Py_False {
		C.__Py_XDECREF(go2c(o))
	}
}

func Py_REFCNT(o *PyObject) int {
	return int(C.__Py_REFCNT(go2c(o)))
}

func Py_CLEAR(o *PyObject) {
	if o != nil && o != Py_None && o != Py_True && o != Py_False {
		C.__Py_CLEAR(go2c(o))
	}
}

func (o *PyObject) Py_Clear() {
	Py_CLEAR(o)
}

func (o *PyObject) Py_RefCnt() int {
	return Py_REFCNT(o)
}

func (o *PyObject) Str() *PyObject {
	return c2go(C.__PyObject_Str(go2c(o)))
}

// int PyObject_HasAttrString(PyObject *o, const char *attr_name)
func (o *PyObject) HasAttrString(name string) bool {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	return int(C.__PyObject_HasAttrString(o.p, cName)) == 1
}

// PyObject* PyObject_GetAttrString(PyObject *o, const char *attr_name)
// Return: new reference.
func (o *PyObject) GetAttrString(name string) *PyObject {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	return c2go(C.__PyObject_GetAttrString(o.p, cName))
}

// int PyObject_SetAttrString(PyObject *o, const char *attr_name, PyObject *v)
func (o *PyObject) SetAttrString(name string, v *PyObject) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	if rc := int(C.__PyObject_SetAttrString(o.p, cName, go2c(v))); rc != 0 {
		return fmt.Errorf("cpython: error to set attribute")
	}
	return nil
}

// int PyCallable_Check(PyObject *o)
func (o *PyObject) Callable() bool {
	return int(C.__PyCallable_Check(go2c(o))) == 1
}

// PyObject* PyObject_Call(PyObject *callable, PyObject *args, PyObject *kwargs)
// Return: new reference.
func (o *PyObject) Call(args, kw *PyObject) *PyObject {
	return c2go(C.__PyObject_Call(o.p, go2c(args), go2c(kw)))
}

// PyObject* PyObject_CallObject(PyObject *callable, PyObject *args)
// Return: new reference.
func (o *PyObject) CallObject(args *PyObject) *PyObject {
	return c2go(C.__PyObject_CallObject(o.p, go2c(args)))
}

func Py_SamePyType(o1, o2 *PyObject) bool {
	return C.__Py_TYPE(go2c(o1)) == C.__Py_TYPE(go2c(o2))
}
