package gvpy

import (
	"fmt"

	"github.com/bluvec/gvpy/python"
)

type GpClass interface {
	GpObject

	New(args ...interface{}) GpInstance

	Call(name string, args ...interface{}) (interface{}, error)
	GetVar(name string) (interface{}, error)
	SetVar(name string, value interface{}) error
}

type gpClass struct {
	gpObject
}

func newGpClass(pyobj *python.PyObject) GpClass {
	c := gpClass{gpObject{pyobj: pyobj}}
	c.setFinalizer()
	return &c
}

// New an instance
func (c *gpClass) New(args ...interface{}) GpInstance {
	i, err := call(c.pyobj, args...)
	if err != nil {
		return nil
	}

	return newGpInstance(i)
}

// Call a class function with trivial arguments and returns.
func (c *gpClass) Call(name string, args ...interface{}) (interface{}, error) {
	f := c.pyobj.GetAttrString(name)
	if f == nil {
		return nil, fmt.Errorf("error to get func '%v' of class '%v'", name, c)
	}
	defer f.Py_Clear()

	retObj, err := call(f, args...)
	if err != nil {
		return nil, fmt.Errorf("error to call func '%v' of class '%v'", name, c)
	}
	defer retObj.Py_Clear()

	return PyToGo(retObj)
}

func (c *gpClass) GetVar(name string) (interface{}, error) {
	return getVar(c.pyobj, name)
}

func (c *gpClass) SetVar(name string, value interface{}) error {
	return setVar(c.pyobj, name, value)
}
