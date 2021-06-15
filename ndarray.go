package gvpy

import (
	"fmt"
	"gvpy/python"
	"runtime"
)

type NpyType int

const (
	NPY_BOOL        NpyType = 0
	NPY_BYTE        NpyType = 1
	NPY_UBYTE       NpyType = 2
	NPY_SHORT       NpyType = 3
	NPY_USHORT      NpyType = 4
	NPY_INT         NpyType = 5
	NPY_UINT        NpyType = 6
	NPY_LONG        NpyType = 7
	NPY_ULONG       NpyType = 8
	NPY_LONGLONG    NpyType = 9
	NPY_ULONGLONG   NpyType = 10
	NPY_FLOAT       NpyType = 11
	NPY_DOUBLE      NpyType = 12
	NPY_LONGDOUBLE  NpyType = 13
	NPY_CFLOAT      NpyType = 14
	NPY_CDOUBLE     NpyType = 15
	NPY_CLONGDOUBLE NpyType = 16
	NPY_OBJECT      NpyType = 17
	NPY_STRING      NpyType = 18
	NPY_UNICODE     NpyType = 19
	NPY_VOID        NpyType = 20
)

type Ndarray interface {
	fmt.Stringer

	Ndim() int
	Shape() []int
	Dtype() NpyType
	Size() int
	AsSlice_byte() []byte
	AsSlice_int() []int
	AsSlice_float64() []float64

	Copy() Ndarray

	// low-level API
	PyObject() *python.PyObject
}

type ndarray struct {
	arr *python.PyObject
}

func setNdarrayFinalizer(ndarr *ndarray) {
	runtime.SetFinalizer(ndarr, func(ndarr *ndarray) { python.Py_XDECREF(ndarr.arr) })
}

// constructors
func NewNdarrayFromPyObject(arr *python.PyObject) Ndarray {
	if arr == nil {
		return nil
	}

	python.Py_XINCREF(arr)
	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarray(shape []int, dtype NpyType) Ndarray {
	arr := python.PyArray_SimpleNew(shape, int(dtype))
	if arr == nil {
		return nil
	}

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayZeros(shape []int, dtype NpyType, fortran bool) Ndarray {
	var arr *python.PyObject
	if fortran {
		arr = python.PyArray_ZEROS(shape, int(dtype), 1)
	} else {
		arr = python.PyArray_ZEROS(shape, int(dtype), 0)
	}
	if arr == nil {
		return nil
	}

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayEmpty(shape []int, dtype NpyType, fortran bool) Ndarray {
	var arr *python.PyObject
	if fortran {
		arr = python.PyArray_EMPTY(shape, int(dtype), 1)
	} else {
		arr = python.PyArray_EMPTY(shape, int(dtype), 0)
	}
	if arr == nil {
		return nil
	}

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayFromSlice_byte(d []byte, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	ndarr := NewNdarray(shape, NPY_BYTE).(*ndarray)
	if ndarr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_byte(ndarr.arr, d)

	return ndarr
}

func NewNdarrayFromSlice_int(d []int, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	ndarr := NewNdarray(shape, NPY_INT).(*ndarray)
	if ndarr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_int(ndarr.arr, d)

	return ndarr
}

func NewNdarrayFromSlice_float64(d []float64, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	ndarr := NewNdarray(shape, NPY_INT).(*ndarray)
	if ndarr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_float64(ndarr.arr, d)

	return ndarr
}

// methods
func (arr *ndarray) String() string {
	return arr.arr.String()
}

func (arr *ndarray) Ndim() int {
	return python.PyArray_NDIM(arr.arr)
}

func (arr *ndarray) Shape() []int {
	return python.PyArray_Shape(arr.arr)
}

func (arr *ndarray) Dtype() NpyType {
	return NpyType(python.PyArray_TYPE(arr.arr))
}

func (arr *ndarray) Size() int {
	return python.PyArray_Size(arr.arr)
}

func (arr *ndarray) AsSlice_byte() []byte {
	return python.PyArray_GetItems_byte(arr.arr)
}

func (arr *ndarray) AsSlice_int() []int {
	return python.PyArray_GetItems_int(arr.arr)
}

func (arr *ndarray) AsSlice_float64() []float64 {
	return python.PyArray_GetItems_float64(arr.arr)
}

func (arr *ndarray) Copy() Ndarray {
	newarr := NewNdarray(arr.Shape(), arr.Dtype()).(*ndarray)
	if newarr == nil {
		return nil
	}

	if err := python.PyArray_Copy(arr.arr, newarr.arr); err != nil {
		return nil
	}

	return newarr
}

func (arr *ndarray) PyObject() *python.PyObject {
	return arr.arr
}

// func ndarrayDimIndToSliceInd(dimInd []int, shape []int) int {
// 	return 0
// }

// func ndarraySliceIndToDimInd(sliceInd int, shape []int) []int {
// 	return nil
// }

func ndarrayCheckSliceLength(sliceLen int, shape []int) int {
	tot := 1
	for idim := range shape {
		tot *= shape[idim]
	}
	if sliceLen < tot {
		return -1
	}
	return tot
}
