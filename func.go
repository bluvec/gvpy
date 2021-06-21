package gvpy

import (
	"fmt"

	"github.com/bluvec/gvpy/python"
)

type GpFunc interface {
	GpObject

	Call(args ...interface{}) (interface{}, error)

	// Lowlevel APIs
	CallX(args ...interface{}) (interface{}, error)
}

type gpFunc struct {
	gpObject
}

func newGpFunc(pyobj *python.PyObject) GpFunc {
	f := gpFunc{gpObject{pyobj: pyobj}}
	f.setFinalizer()
	return &f
}

// Call a function.
func (f *gpFunc) Call(args ...interface{}) (interface{}, error) {
	gil := GILEnsure()
	defer GILRelease(gil)

	return f.CallX(args...)
}

// Lowlevel API: call a function.
func (f *gpFunc) CallX(args ...interface{}) (interface{}, error) {
	retObj, err := call(f.pyobj, args...)
	if err != nil {
		return nil, fmt.Errorf("error to call func '%v', err: '%v'", f, err)
	}
	defer retObj.Py_Clear()

	return pyToGo(retObj)
}
