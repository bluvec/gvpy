package gvpy

import (
	"fmt"
	"gvpy/python"
)

type GpModule struct {
	pyobj *python.PyObject
}

func Import(name string) (*GpModule, error) {
	if m := python.PyImport_ImportModule(name); m == nil {
		return nil, fmt.Errorf("error to import module '%v'", name)
	} else {
		return &GpModule{pyobj: m}, nil
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

func FromImport(from, name string) (*GpModule, error) {
	m, err := fromImport(from, name)
	if err != nil {
		return nil, err
	}

	if !python.PyModule_Check(m) {
		defer m.Py_Clear()
		return nil, fmt.Errorf("error to import: %v is not a module", m)
	}

	return &GpModule{pyobj: m}, nil
}

func FromImportFunc(from, name string) (*GpFunc, error) {
	f, err := fromImport(from, name)
	if err != nil {
		return nil, err
	}

	if !python.PyFunction_Check(f) {
		defer f.Py_Clear()
		return nil, fmt.Errorf("error to import: %v is not a function", f)
	}

	return &GpFunc{pyobj: f}, nil
}

func FromImportClass(from, name string) (*GpClass, error) {
	c, err := fromImport(from, name)
	if err != nil {
		return nil, err
	}
	if !python.PyClass_Check(c) {
		defer c.Py_Clear()
		return nil, fmt.Errorf("error to import: %v is not a class", c)
	}

	return &GpClass{pyobj: c}, err
}

func (m *GpModule) String() string {
	return m.pyobj.String()
}

func (m *GpModule) Clear() {
	m.pyobj.Py_Clear()
}

func (m *GpModule) GetVar(name string) (interface{}, error) {
	return getVar(m.pyobj, name)
}

func (m *GpModule) SetVar(name string, value interface{}) error {
	return setVar(m.pyobj, name, value)
}

func (m *GpModule) GetClassVar(class, name string) (interface{}, error) {
	panic("not implemented")
}

func (m *GpModule) SetClassVar(class, name string, value interface{}) error {
	panic("not implemented")
}

func (m *GpModule) GetFunc(name string) (*GpFunc, error) {
	if f := m.pyobj.GetAttrString(name); f == nil {
		return nil, fmt.Errorf("error to get func '%v.%v'", m, name)
	} else {
		return &GpFunc{pyobj: f}, nil
	}
}

func (m *GpModule) CallFunc(name string, args ...interface{}) (interface{}, error) {
	gf, err := m.GetFunc(name)
	if err != nil {
		return nil, err
	}
	defer gf.Clear()

	return gf.Call(args...)
}

func (m *GpModule) GetClass(name string) (*GpClass, error) {
	if c := m.pyobj.GetAttrString(name); c == nil {
		return nil, fmt.Errorf("error to get class '%v.%v'", m, name)
	} else {
		return &GpClass{pyobj: c}, nil
	}
}

// New an instance of a class
func (m *GpModule) New(class string, args ...interface{}) (*GpInstance, error) {
	gc, err := m.GetClass(class)
	if err != nil {
		return nil, err
	}
	defer gc.Clear()

	return gc.New(args...)
}
