package gvpy

import (
	"github.com/bluvec/gvpy/python"
)

type GpInstance interface {
	GpObject

	Call(methName string, args ...interface{}) (interface{}, error)
	GetVar(name string) (interface{}, error)
	SetVar(name string, value interface{}) error
}

type gpInstance struct {
	gpObject
}

func newGpInstance(pyobj *python.PyObject) GpInstance {
	i := gpInstance{gpObject{pyobj: pyobj}}
	i.setFinalizer()
	return &i
}

// Call an instance method
func (i *gpInstance) Call(methName string, args ...interface{}) (interface{}, error) {
	methObj, err := getAttr(i.pyobj, methName)
	if err != nil {
		return nil, err
	}
	defer methObj.Py_Clear()

	retObj, err := call(methObj, args...)
	if err != nil {
		return nil, err
	}
	defer retObj.Py_Clear()

	return PyToGo(retObj)
}

func (i *gpInstance) GetVar(name string) (interface{}, error) {
	return getVar(i.pyobj, name)
}

func (i *gpInstance) SetVar(name string, value interface{}) error {
	return setVar(i.pyobj, name, value)
}
