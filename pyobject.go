package gvpy

import "github.com/bluvec/gvpy/python"

type PyObjectGetter interface {
	PyObject() *python.PyObject
}
