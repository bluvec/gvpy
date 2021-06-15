package gvpy

import (
	"fmt"
	"gvpy/python"
)

type GpClass struct {
	pyobj *python.PyObject
}

func (c *GpClass) String() string {
	return c.pyobj.String()
}

func (c *GpClass) Clear() {
	c.pyobj.Py_Clear()
}

// New an instance
func (c *GpClass) New(args ...interface{}) (*GpInstance, error) {
	i, err := call(c.pyobj, args...)
	if err != nil {
		return nil, fmt.Errorf("error to new instance of class '%v', err: '%v'", c, err)
	}
	return &GpInstance{pyobj: i}, nil
}

// Call a class function with trivial arguments and returns.
func (c *GpClass) Call(name string, args ...interface{}) (interface{}, error) {
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

func (c *GpClass) GetVar(name string) (interface{}, error) {
	return getVar(c.pyobj, name)
}

func (c *GpClass) SetVar(name string, value interface{}) error {
	return setVar(c.pyobj, name, value)
}
