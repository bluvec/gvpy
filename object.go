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
	Clear()

	setFinalizer()
}

type gpObject struct {
	pyobj *python.PyObject
}

func (o *gpObject) RefCnt() int {
	return o.pyobj.Py_RefCnt()
}

func (o *gpObject) Clear() {
	pyobj := atomic.SwapPointer((*unsafe.Pointer)(unsafe.Pointer(&o.pyobj)), nil)
	if pyobj != nil {
		(*python.PyObject)(pyobj).Py_Clear()
	}
}

func (o *gpObject) String() string {
	return o.pyobj.String()
}

func (o *gpObject) setFinalizer() {
	runtime.SetFinalizer(o, func(o *gpObject) {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		gil := GILEnsure()
		defer GILRelease(gil)

		o.Clear()
	})
}
