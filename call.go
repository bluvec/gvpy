package gvpy

import (
	"fmt"
	"gvpy/python"
)

func call(callable *python.PyObject, args ...interface{}) (*python.PyObject, error) {
	// Prepare args
	pyTuple, err := trivialArgsToPyTuple(args...)
	if err != nil {
		return nil, err
	}
	defer pyTuple.Py_Clear()

	// Call python callable object
	retObj := callable.Call(pyTuple, nil)

	// Organize returns
	if retObj == nil {
		return nil, fmt.Errorf("error to call python object: %v", callable)
	}

	return retObj, nil
}

func trivialArgsToPyTuple(args ...interface{}) (*python.PyObject, error) {
	nArgs := len(args)

	// Create PyTuple
	pyTuple := python.PyTuple_New(nArgs)
	if pyTuple == nil {
		return nil, fmt.Errorf("error to create PyTuple with size '%v'", nArgs)
	}

	// Create trivial args to PyObject
	for i := 0; i < nArgs; i++ {
		if pyarg, err := GoToPy(args[i]); err != nil {
			return nil, err
		} else if err := python.PyTuple_SetItem(pyTuple, i, pyarg); err != nil {
			return nil, err
		}
	}

	return pyTuple, nil
}
