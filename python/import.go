package python

// #include "py.h"
import "C"
import (
	"unsafe"
)

// PyObject* PyImport_ImportModule(const char *name)
// Return: new reference.
func PyImport_ImportModule(name string) *PyObject {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	return c2go(C.__PyImport_ImportModule(cName))
}

// PyObject* PyImport_ImportModuleEx(const char *name, PyObject *globals, PyObject *locals, PyObject *fromlist)
// Return: new reference.
func PyImport_ImportModuleEx(name string, globals, locals, fromlist *PyObject) *PyObject {
	return PyImport_ImportModuleLevel(name, globals, locals, fromlist, 0)
}

// PyObject* PyImport_ImportModuleLevel(const char *name, PyObject *globals, PyObject *locals, PyObject *fromlist, int level)
// Return: new reference.
func PyImport_ImportModuleLevel(name string, globals, locals, fromlist *PyObject, level int) *PyObject {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	return c2go(C.__PyImport_ImportModuleLevel(cName, go2c(globals), go2c(locals), go2c(fromlist), C.int(level)))
}
