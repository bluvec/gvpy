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

	// Lowlevel APIs
	GetVarX(name string) (interface{}, error)
	SetVarX(name string, value interface{}) error
	GetClassVarX(class, name string) (interface{}, error)
	SetClassVarX(class, name string, value interface{}) error
	GetFuncX(name string) GpFunc
	CallFuncX(name string, args ...interface{}) (interface{}, error)
	GetClassX(name string) GpClass
	NewX(class string, args ...interface{}) GpInstance
}

type gpModule struct {
	gpObject
}

func newGpModule(pyobj *python.PyObject) GpModule {
	m := gpModule{gpObject{pyobj: pyobj}}
	m.setFinalizer()
	return &m
}

// Import a module.
func Import(name string) GpModule {
	gil := GILEnsure()
	defer GILRelease(gil)

	return ImportX(name)
}

// From a module import a submodule.
func FromImport(from, name string) GpModule {
	gil := GILEnsure()
	defer GILRelease(gil)

	return FromImportX(from, name)
}

// From a module to import a function.
func FromImportFunc(from, name string) GpFunc {
	gil := GILEnsure()
	defer GILRelease(gil)

	return FromImportFuncX(from, name)
}

// From a module to import a class.
func FromImportClass(from, name string) GpClass {
	gil := GILEnsure()
	defer GILRelease(gil)

	return FromImportClassX(from, name)
}

// Get module variable.
func (m *gpModule) GetVar(name string) (interface{}, error) {
	gil := GILEnsure()
	defer GILRelease(gil)

	return m.GetVarX(name)
}

// Set module variable.
func (m *gpModule) SetVar(name string, value interface{}) error {
	gil := GILEnsure()
	defer GILRelease(gil)

	return m.SetVarX(name, value)
}

// Get class variable.
func (m *gpModule) GetClassVar(class, name string) (interface{}, error) {
	gil := GILEnsure()
	defer GILRelease(gil)

	return m.GetClassVarX(class, name)
}

// Set class variable.
func (m *gpModule) SetClassVar(class, name string, value interface{}) error {
	gil := GILEnsure()
	defer GILRelease(gil)

	return m.SetClassVarX(class, name, value)
}

// Get a function.
func (m *gpModule) GetFunc(name string) GpFunc {
	gil := GILEnsure()
	defer GILRelease(gil)

	return m.GetFuncX(name)
}

// Call a function.
func (m *gpModule) CallFunc(name string, args ...interface{}) (interface{}, error) {
	gil := GILEnsure()
	defer GILRelease(gil)

	return m.CallFuncX(name, args...)
}

// Get a class.
func (m *gpModule) GetClass(name string) GpClass {
	gil := GILEnsure()
	defer GILRelease(gil)

	return m.GetClassX(name)
}

// New an instance of a class.
func (m *gpModule) New(class string, args ...interface{}) GpInstance {
	gil := GILEnsure()
	defer GILRelease(gil)

	return m.NewX(class, args...)
}

// Lowlevel API: import a module.
func ImportX(name string) GpModule {
	if m := python.PyImport_ImportModule(name); m == nil {
		return nil
	} else {
		return newGpModule(m)
	}
}

// Lowlevel API: from a module import a submodule.
func FromImportX(from, name string) GpModule {
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

// Lowlevel API: from a module import a function.
func FromImportFuncX(from, name string) GpFunc {
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

// Lowlevel API: from a module import a class.
func FromImportClassX(from, name string) GpClass {
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

// Lowlevel API: get module variable.
func (m *gpModule) GetVarX(name string) (interface{}, error) {
	return getVar(m.pyobj, name)
}

// Lowlevel API: set module variable.
func (m *gpModule) SetVarX(name string, value interface{}) error {
	return setVar(m.pyobj, name, value)
}

// Lowlevel API: get a class.
func (m *gpModule) GetClassVarX(class, name string) (interface{}, error) {
	panic("not implemented")
}

// Lowlevel API: set class variable.
func (m *gpModule) SetClassVarX(class, name string, value interface{}) error {
	panic("not implemented")
}

// Lowlevel API: get a function.
func (m *gpModule) GetFuncX(name string) GpFunc {
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

// Lowlevel API: call a function.
func (m *gpModule) CallFuncX(name string, args ...interface{}) (interface{}, error) {
	f := m.GetFuncX(name)
	if f == nil {
		return nil, fmt.Errorf("error to get func %v", name)
	}

	return f.CallX(args...)
}

// Lowlevel API: get a class.
func (m *gpModule) GetClassX(name string) GpClass {
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

// Lowlevel API: new an instance of a class.
func (m *gpModule) NewX(class string, args ...interface{}) GpInstance {
	c := m.GetClassX(class)
	if c == nil {
		return nil
	}

	return c.NewX(args...)
}

// Private functions
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
