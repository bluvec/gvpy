package gvpy

import (
	"fmt"

	"github.com/bluvec/gvpy/python"
)

type GpModule interface {
	GpObject

	GetVar(name string) (interface{}, error)
	SetVar(name string, value interface{}) error
	GetClassVar(class, name string) (interface{}, error)
	SetClassVar(class, name string, value interface{}) error
	GetFunc(name string) GpFunc
	CallFunc(name string, args ...interface{}) (interface{}, error)
	GetClass(name string) GpClass
	New(class string, args ...interface{}) GpInstance
}

type gpModule struct {
	gpObject
}

func newGpModule(pyobj *python.PyObject) GpModule {
	m := gpModule{gpObject{pyobj: pyobj}}
	m.setFinalizer()
	return &m
}

func Import(name string) GpModule {
	if m := python.PyImport_ImportModule(name); m == nil {
		return nil
	} else {
		return newGpModule(m)
	}
}

func fromImport(from, name string) (*python.PyObject, error) {
	fromlist, err := trivialArgsToPyTuple(name)
	if err != nil {
		return nil, err
	}
	defer fromlist.Py_Clear()

	tmp := python.PyImport_ImportModuleEx(from, nil, nil, fromlist)
	if tmp == nil {
		return nil, fmt.Errorf("error to import: `from %v import %v`", from, name)
	}
	defer tmp.Py_Clear()

	m := tmp.GetAttrString(name)
	if m == nil {
		return nil, fmt.Errorf("error to import: 'from %v import %v'", from, name)
	}

	return m, nil
}

func FromImport(from, name string) GpModule {
	m, err := fromImport(from, name)
	if err != nil {
		return nil
	}

	if !python.PyModule_Check(m) {
		defer m.Py_Clear()
		return nil
	}

	return newGpModule(m)
}

func FromImportFunc(from, name string) GpFunc {
	f, err := fromImport(from, name)
	if err != nil {
		return nil
	}

	if !python.PyFunction_Check(f) {
		defer f.Py_Clear()
		return nil
	}

	return newGpFunc(f)
}

func FromImportClass(from, name string) GpClass {
	c, err := fromImport(from, name)
	if err != nil {
		return nil
	}

	if !python.PyClass_Check(c) {
		defer c.Py_Clear()
		return nil
	}

	return newGpClass(c)
}

func (m *gpModule) GetVar(name string) (interface{}, error) {
	return getVar(m.pyobj, name)
}

func (m *gpModule) SetVar(name string, value interface{}) error {
	return setVar(m.pyobj, name, value)
}

func (m *gpModule) GetClassVar(class, name string) (interface{}, error) {
	panic("not implemented")
}

func (m *gpModule) SetClassVar(class, name string, value interface{}) error {
	panic("not implemented")
}

func (m *gpModule) GetFunc(name string) GpFunc {
	pyobj := m.pyobj.GetAttrString(name)
	if pyobj == nil {
		return nil
	}

	if !python.PyFunction_Check(pyobj) {
		defer pyobj.Py_Clear()
		return nil
	}

	return newGpFunc(pyobj)
}

func (m *gpModule) CallFunc(name string, args ...interface{}) (interface{}, error) {
	f := m.GetFunc(name)
	if f == nil {
		return nil, fmt.Errorf("error to get func %v", name)
	}

	return f.Call(args...)
}

func (m *gpModule) GetClass(name string) GpClass {
	pyobj := m.pyobj.GetAttrString(name)
	if pyobj == nil {
		return nil
	}

	if !python.PyClass_Check(pyobj) {
		defer pyobj.Py_Clear()
		return nil
	}

	return newGpClass(pyobj)
}

// New an instance of a class
func (m *gpModule) New(class string, args ...interface{}) GpInstance {
	c := m.GetClass(class)
	if c == nil {
		return nil
	}

	return c.New(args...)
}
