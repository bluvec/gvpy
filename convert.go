package gvpy

import (
	"fmt"
	"reflect"

	"github.com/bluvec/gvpy/python"
)

type GoToPyConv func(arg interface{}) (*python.PyObject, error)
type PyToGoConv func(pyobj *python.PyObject) (interface{}, error)

var gGoToPy []GoToPyConv
var gPyToGo []PyToGoConv

func RegGoToPyConv(conv GoToPyConv) {
	gGoToPy = append(gGoToPy, conv)
}

func RegPyToGoConv(conv PyToGoConv) {
	gPyToGo = append(gPyToGo, conv)
}

func GoToPy(arg interface{}) (*python.PyObject, error) {
	switch val := arg.(type) {
	// none
	case nil:
		return python.Py_None, nil

	// trivial
	case bool:
		if val {
			return python.Py_True, nil
		} else {
			return python.Py_False, nil
		}
	case int:
		return python.PyLong_FromLong(val), nil
	case float32:
		return python.PyFloat_FromDouble(float64(val)), nil
	case float64:
		return python.PyFloat_FromDouble(val), nil
	case string:
		return python.PyUnicode_FromString(val), nil
	case []byte:
		return python.PyBytes_FromBytes(val), nil

	// numpy.ndarray
	case *ndarray:
		python.Py_XINCREF(val.arr)
		return val.arr, nil

	// complex combinations
	default:
		argKind := reflect.TypeOf(arg).Kind()
		if argKind == reflect.Array || argKind == reflect.Slice {
			// list
			varg := reflect.ValueOf(arg)
			nele := varg.Len()

			pyList := python.PyList_New(nele)
			if pyList == nil {
				return nil, fmt.Errorf("error to PyList_New with len '%v'", nele)
			}

			for i := 0; i < nele; i++ {
				if pv, err := GoToPy(varg.Index(i).Interface()); err != nil {
					return nil, err
				} else {
					python.PyList_SetItem(pyList, i, pv)
				}
			}

			return pyList, nil
		} else if argKind == reflect.Map {
			// dict
			pdict := python.PyDict_New()
			if pdict == nil {
				return nil, fmt.Errorf("error to PyDict_new")
			}

			varg := reflect.ValueOf(arg)
			iter := varg.MapRange()
			for iter.Next() {
				k := iter.Key().Interface()
				v := iter.Value().Interface()

				pk, err := GoToPy(k)
				if err != nil {
					return nil, err
				}

				pv, err := GoToPy(v)

				if err != nil {
					return nil, err
				}

				python.PyDict_SetItem(pdict, pk, pv)
			}

			return pdict, nil
		} else {
			for i := range gGoToPy {
				if pyobj, err := gGoToPy[i](arg); err == nil {
					return pyobj, nil
				}
			}
		}
	}

	return nil, fmt.Errorf("unsupported type: %+v", reflect.TypeOf(arg))
}

func PyToGo(pyobj *python.PyObject) (interface{}, error) {
	if pyobj == python.Py_None {
		return nil, nil
	} else if python.PyBool_Check(pyobj) {
		return python.PyBool_IsTure(pyobj), nil
	} else if python.PyLong_Check(pyobj) {
		return python.PyLong_AsLong(pyobj), nil
	} else if python.PyFloat_Check(pyobj) {
		return python.PyFloat_AsDouble(pyobj), nil
	} else if python.PyBytes_Check(pyobj) {
		return python.PyBytes_AsBytes(pyobj), nil
	} else if python.PyUnicode_Check(pyobj) {
		return python.PyUnicode_AsString(pyobj), nil
	} else if python.PyArray_Check(pyobj) {
		return NewNdarrayFromPyObject(pyobj), nil
	} else if python.PyList_Check(pyobj) {
		sz := python.PyList_Size(pyobj)
		if sz == 0 {
			return make([]interface{}, 0), nil
		}

		if isTrivialPyList(pyobj) {
			return pyobjectToTrivialList(pyobj)
		} else {
			return pyobjectToNonTrivialList(pyobj)
		}
	} else if python.PyTuple_Check(pyobj) {
		sz := python.PyTuple_Size(pyobj)
		if sz == 0 {
			return make([]interface{}, 0), nil
		}

		return pyobjectToNonTrivialTuple(pyobj)
	} else if python.PyDict_Check(pyobj) {
		sz := python.PyDict_Size(pyobj)
		if sz == 0 {
			return make(map[interface{}]interface{}), nil
		}

		if isTrivialPyDict(pyobj) {
			return pyobjectToTrivialDict(pyobj)
		} else {
			return pyobjectToNonTrivialDict(pyobj)
		}
	} else {
		for i := range gPyToGo {
			if obj, err := gPyToGo[i](pyobj); err == nil {
				return obj, nil
			}
		}
	}

	return nil, fmt.Errorf("unknown PyObject type")
}

func isTrivialPyObject(pyobj *python.PyObject) bool {
	return python.PyBool_Check(pyobj) || python.PyLong_Check(pyobj) || python.PyFloat_Check(pyobj) || python.PyUnicode_Check(pyobj)
}

func isTrivialPyList(pyobj *python.PyObject) bool {
	sz := python.PyList_Size(pyobj)

	v0 := python.PyList_GetItem(pyobj, 0)
	if !isTrivialPyObject(v0) {
		return false
	}

	for i := 1; i < sz; i++ {
		v := python.PyList_GetItem(pyobj, i)
		if !python.Py_SamePyType(v0, v) {
			return false
		}
	}

	return true
}

func isTrivialPyDict(pyobj *python.PyObject) bool {
	var pk0, pv0 *python.PyObject
	var pos int = 0

	python.PyDict_Next(pyobj, &pos, &pk0, &pv0)

	if !python.PyUnicode_Check(pk0) {
		return false
	}
	if !isTrivialPyObject(pv0) {
		return false
	}

	var pk, pv *python.PyObject
	for python.PyDict_Next(pyobj, &pos, &pk, &pv) {
		if !python.Py_SamePyType(pk0, pk) {
			return false
		}

		if !python.Py_SamePyType(pv0, pv) {
			return false
		}
	}

	return true
}

func pyobjectToTrivialList(pyobj *python.PyObject) (interface{}, error) {
	sz := python.PyList_Size(pyobj)
	v0 := python.PyList_GetItem(pyobj, 0)

	if python.PyBool_Check(v0) {
		list := make([]bool, sz)
		for i := 0; i < sz; i++ {
			vi := python.PyList_GetItem(pyobj, i)
			v := python.PyBool_IsTure(vi)
			list[i] = v
		}
		return list, nil
	} else if python.PyLong_Check(v0) {
		list := make([]int, sz)
		for i := 0; i < sz; i++ {
			vi := python.PyList_GetItem(pyobj, i)
			v := python.PyLong_AsLong(vi)
			list[i] = v
		}
		return list, nil
	} else if python.PyFloat_Check(v0) {
		list := make([]float64, sz)
		for i := 0; i < sz; i++ {
			vi := python.PyList_GetItem(pyobj, i)
			v := python.PyFloat_AsDouble(vi)
			list[i] = v
		}
		return list, nil
	} else if python.PyUnicode_Check(v0) {
		list := make([]string, sz)
		for i := 0; i < sz; i++ {
			vi := python.PyList_GetItem(pyobj, i)
			v := python.PyUnicode_AsString(vi)
			list[i] = v
		}
		return list, nil
	}

	return nil, fmt.Errorf("unsupported PyList item type")
}

func pyobjectToNonTrivialList(pyobj *python.PyObject) (interface{}, error) {
	sz := python.PyList_Size(pyobj)
	list := make([]interface{}, sz)
	for i := 0; i < sz; i++ {
		vi := python.PyList_GetItem(pyobj, i)
		v, err := PyToGo(vi)
		if err != nil {
			return nil, err
		}
		list[i] = v
	}
	return list, nil
}

func pyobjectToNonTrivialTuple(pyobj *python.PyObject) (interface{}, error) {
	sz := python.PyTuple_Size(pyobj)
	tuple := make([]interface{}, sz)
	for i := 0; i < sz; i++ {
		vi := python.PyTuple_GetItem(pyobj, i)
		v, err := PyToGo(vi)
		if err != nil {
			return nil, err
		}
		tuple[i] = v
	}
	return tuple, nil
}

func pyobjectToTrivialDict(pyobj *python.PyObject) (interface{}, error) {
	var pk0, pv0 *python.PyObject
	var pos int = 0

	python.PyDict_Next(pyobj, &pos, &pk0, &pv0)

	if python.PyBool_Check(pv0) {
		dict := make(map[string]bool)
		k := python.PyUnicode_AsString(pk0)
		v := python.PyBool_IsTure(pv0)
		dict[k] = v

		var pk, pv *python.PyObject
		for python.PyDict_Next(pyobj, &pos, &pk, &pv) {
			k := python.PyUnicode_AsString(pk)
			v := python.PyBool_IsTure(pv)
			dict[k] = v
		}
		return dict, nil
	} else if python.PyLong_Check(pv0) {
		dict := make(map[string]int)
		k := python.PyUnicode_AsString(pk0)
		v := python.PyLong_AsLong(pv0)
		dict[k] = v

		var pk, pv *python.PyObject
		for python.PyDict_Next(pyobj, &pos, &pk, &pv) {
			k := python.PyUnicode_AsString(pk)
			v := python.PyLong_AsLong(pv)
			dict[k] = v
		}
		return dict, nil
	} else if python.PyFloat_Check(pv0) {
		dict := make(map[string]float64)
		k := python.PyUnicode_AsString(pk0)
		v := python.PyFloat_AsDouble(pv0)
		dict[k] = v

		var pk, pv *python.PyObject
		for python.PyDict_Next(pyobj, &pos, &pk, &pv) {
			k := python.PyUnicode_AsString(pk)
			v := python.PyFloat_AsDouble(pv)
			dict[k] = v
		}
		return dict, nil
	} else if python.PyUnicode_Check(pv0) {
		dict := make(map[string]string)
		k := python.PyUnicode_AsString(pk0)
		v := python.PyUnicode_AsString(pv0)
		dict[k] = v

		var pk, pv *python.PyObject
		for python.PyDict_Next(pyobj, &pos, &pk, &pv) {
			k := python.PyUnicode_AsString(pk)
			v := python.PyUnicode_AsString(pv)
			dict[k] = v
		}
		return dict, nil
	}

	return nil, fmt.Errorf("unsupported PyDict key/value type")
}

func pyobjectToNonTrivialDict(pyobj *python.PyObject) (interface{}, error) {
	dict := make(map[interface{}]interface{})

	var pk, pv *python.PyObject
	var pos int = 0

	for python.PyDict_Next(pyobj, &pos, &pk, &pv) {
		k, err := PyToGo(pk)
		if err != nil {
			return nil, err
		}
		v, err := PyToGo(pv)
		if err != nil {
			return nil, err
		}

		dict[k] = v
	}

	return dict, nil
}
