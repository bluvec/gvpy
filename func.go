package gvpy

import (
	"fmt"
	"gvpy/python"
)

type GpFunc struct {
	pyobj *python.PyObject
}

func (f *GpFunc) String() string {
	return f.pyobj.String()
}

func (f *GpFunc) Clear() {
	f.pyobj.Py_Clear()
}

// Call a function with trivial arguments and returns.
func (f *GpFunc) Call(args ...interface{}) (interface{}, error) {
	retObj, err := call(f.pyobj, args...)
	if err != nil {
		return nil, fmt.Errorf("error to call func '%v', err: '%v'", f, err)
	}
	defer retObj.Py_Clear()

	return PyToGo(retObj)
}
