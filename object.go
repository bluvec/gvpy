package gvpy

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"unsafe"

	"github.com/bluvec/gvpy/python"
)

type GpObject interface {
	fmt.Stringer

	RefCnt() int
	Close()

	// Lowlevel APIs
	RefCntX() int
	CloseX()
}

type gpObject struct {
	pyobj *python.PyObject
}

// Get the reference count of a python object.
func (o *gpObject) RefCnt() int {
	gil := GILEnsure()
	defer GILRelease(gil)

	return o.RefCntX()
}

// Close the python object. This function is not necessary to call.
// The golang GC will recycle the unreferenced python object automatically.
func (o *gpObject) Close() {
	pyobj := atomic.SwapPointer((*unsafe.Pointer)(unsafe.Pointer(&o.pyobj)), nil)
	if pyobj != nil {
		gil := GILEnsure()
		(*python.PyObject)(pyobj).Py_Clear()
		GILRelease(gil)
	}
}

func (o *gpObject) String() string {
	gil := GILEnsure()
	defer GILRelease(gil)

	return o.pyobj.String()
}

func (o *gpObject) setFinalizer() {
	runtime.SetFinalizer(o, func(o *gpObject) { o.Close() })
}

// Lowlevel APIs
func (o *gpObject) RefCntX() int {
	return o.pyobj.Py_RefCnt()
}

func (o *gpObject) CloseX() {
	pyobj := atomic.SwapPointer((*unsafe.Pointer)(unsafe.Pointer(&o.pyobj)), nil)
	if pyobj != nil {
		(*python.PyObject)(pyobj).Py_Clear()
	}
}
