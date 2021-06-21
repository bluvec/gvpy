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

	// Lowlevel APIs
	NewX(args ...interface{}) GpInstance
	CallX(name string, args ...interface{}) (interface{}, error)
	GetVarX(name string) (interface{}, error)
	SetVarX(name string, value interface{}) error
}

type gpClass struct {
	gpObject
}

func newGpClass(pyobj *python.PyObject) GpClass {
	c := gpClass{gpObject{pyobj: pyobj}}
	c.setFinalizer()
	return &c
}

// New an instance.
func (c *gpClass) New(args ...interface{}) GpInstance {
	gil := GILEnsure()
	defer GILRelease(gil)

	return c.NewX(args...)
}

// Call a class function, e.g., the static method.
func (c *gpClass) Call(name string, args ...interface{}) (interface{}, error) {
	gil := GILEnsure()
	defer GILRelease(gil)

	return c.CallX(name, args...)
}

// Get a class variable.
func (c *gpClass) GetVar(name string) (interface{}, error) {
	gil := GILEnsure()
	defer GILRelease(gil)

	return c.GetVarX(name)
}

// Set a class variable.
func (c *gpClass) SetVar(name string, value interface{}) error {
	gil := GILEnsure()
	defer GILRelease(gil)

	return c.SetVarX(name, value)
}

// Lowlevel API: new an instance
func (c *gpClass) NewX(args ...interface{}) GpInstance {
	i, err := call(c.pyobj, args...)
	if err != nil {
		return nil
	}

	return newGpInstance(i)
}

// Lowlevel API: Call a class function.
func (c *gpClass) CallX(name string, args ...interface{}) (interface{}, error) {
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

	return pyToGo(retObj)
}

// Lowlevel API: get a class variable.
func (c *gpClass) GetVarX(name string) (interface{}, error) {
	return getVar(c.pyobj, name)
}

// Lowlevel API: set a class variable.
func (c *gpClass) SetVarX(name string, value interface{}) error {
	return setVar(c.pyobj, name, value)
}
