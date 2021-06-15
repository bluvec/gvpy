package gvpy

import (
	"fmt"
	"gvpy/python"
)

func getAttr(pyobj *python.PyObject, name string) (*python.PyObject, error) {
	attrObj := pyobj.GetAttrString(name)
	if attrObj == nil {
		return nil, fmt.Errorf("error to get attr '%v' from '%v'", name, pyobj)
	}

	return attrObj, nil
}

func getVar(pyobj *python.PyObject, name string) (interface{}, error) {
	attrObj, err := getAttr(pyobj, name)
	if err != nil {
		return nil, err
	}
	defer attrObj.Py_Clear()

	return PyToGo(attrObj)
}

func setAttr(pyobj *python.PyObject, name string, pyval *python.PyObject) error {
	return pyobj.SetAttrString(name, pyval)
}

func setVar(pyobj *python.PyObject, name string, value interface{}) error {
	pyval, err := GoToPy(value)
	if err != nil {
		return err
	}
	defer pyval.Py_Clear()

	return setAttr(pyobj, name, pyval)
}
