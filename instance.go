package gvpy

import (
	"fmt"
	"gvpy/python"
)

type GpInstance struct {
	pyobj *python.PyObject
}

func (i *GpInstance) String() string {
	return i.pyobj.String()
}

func (i *GpInstance) Clear() {
	i.pyobj.Py_Clear()
}

// Call an instance method
func (i *GpInstance) Call(methName string, args ...interface{}) (interface{}, error) {
	methObj, err := getAttr(i.pyobj, methName)
	if err != nil {
		return nil, fmt.Errorf("error to call method '%v.%v', err: '%v'", i, methName, err)
	}
	defer methObj.Py_Clear()

	retObj, err := call(methObj, args...)
	if err != nil {
		return nil, err
	}
	defer retObj.Py_Clear()

	return PyToGo(retObj)
}

func (i *GpInstance) GetVar(name string) (interface{}, error) {
	return getVar(i.pyobj, name)
}

func (i *GpInstance) SetVar(name string, value interface{}) error {
	return setVar(i.pyobj, name, value)
}
