package python

// #include "py.h"
import "C"
import (
	"unsafe"
)

// void Py_Initialize()
func Py_Initialize() {
	C.__Py_Initialize()
}

// int Py_IsInitialized()
func Py_IsInitialized() bool {
	return int(C.__Py_IsInitialized()) != 0
}

// void Py_Finalize()
func Py_Finalize() {
	C.__Py_Finalize()
}

// int Py_FinalizeEx()
func Py_FinalizeEx() int {
	return int(C.__Py_FinalizeEx())
}

// void PyEval_InitThreads()
func PyEval_InitThreads() {
	C.__PyEval_InitThreads()
}

// int PyEval_ThreadsInitialized()
func PyEval_ThreadsInitialized() bool {
	return int(C.__PyEval_ThreadsInitialized()) != 0
}

func Py_GetPath() string {
	cpath := C.__Py_GetPath()
	defer C.__PyMem_Free(unsafe.Pointer(cpath))

	return C.GoString(cpath)
}

func Py_SetPath(path string) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	C.__Py_SetPath(cpath)
}

func Py_GetVersion() string {
	cVer := C.__Py_GetVersion()
	return C.GoString(cVer)
}
