package gvpy

import (
	"fmt"
	"gvpy/python"
	"runtime"
)

type NpyType = python.NpyType

const (
	NPY_BOOL        NpyType = python.NPY_BOOL
	NPY_BYTE        NpyType = python.NPY_BYTE
	NPY_UBYTE       NpyType = python.NPY_UBYTE
	NPY_SHORT       NpyType = python.NPY_SHORT
	NPY_USHORT      NpyType = python.NPY_USHORT
	NPY_INT         NpyType = python.NPY_INT
	NPY_UINT        NpyType = python.NPY_UINT
	NPY_LONG        NpyType = python.NPY_LONG
	NPY_ULONG       NpyType = python.NPY_ULONG
	NPY_LONGLONG    NpyType = python.NPY_LONGLONG
	NPY_ULONGLONG   NpyType = python.NPY_ULONGLONG
	NPY_FLOAT       NpyType = python.NPY_FLOAT
	NPY_DOUBLE      NpyType = python.NPY_DOUBLE
	NPY_LONGDOUBLE  NpyType = python.NPY_LONGDOUBLE
	NPY_CFLOAT      NpyType = python.NPY_CFLOAT
	NPY_CDOUBLE     NpyType = python.NPY_CDOUBLE
	NPY_CLONGDOUBLE NpyType = python.NPY_CLONGDOUBLE
	NPY_OBJECT      NpyType = python.NPY_OBJECT
	NPY_STRING      NpyType = python.NPY_STRING
	NPY_UNICODE     NpyType = python.NPY_UNICODE
	NPY_VOID        NpyType = python.NPY_VOID

	NPY_INT8       NpyType = NPY_BYTE
	NPY_UINT8      NpyType = NPY_UBYTE
	NPY_INT16      NpyType = NPY_SHORT
	NPY_UINT16     NpyType = NPY_USHORT
	NPY_INT32      NpyType = NPY_INT
	NPY_UINT32     NpyType = NPY_UINT
	NPY_INT64      NpyType = NPY_LONG
	NPY_UINT64     NpyType = NPY_ULONG
	NPY_FLOAT32    NpyType = NPY_FLOAT
	NPY_FLOAT64    NpyType = NPY_DOUBLE
	NPY_COMPLEX64  NpyType = NPY_CFLOAT
	NPY_COMPLEX128 NpyType = NPY_CDOUBLE
)

type Ndarray interface {
	fmt.Stringer

	Ndim() int
	Shape() []int
	Dtype() NpyType
	Size() int
	Flags() int
	IsFortran() bool

	AsSlice_byte() []byte
	AsSlice_int() []int
	AsSlice_uint() []uint
	AsSlice_int8() []int8
	AsSlice_uint8() []uint8
	AsSlice_int16() []int16
	AsSlice_uint16() []uint16
	AsSlice_int32() []int32
	AsSlice_uint32() []uint32
	AsSlice_int64() []int64
	AsSlice_uint64() []uint64
	AsSlice_complex64() []complex64
	AsSlice_complex128() []complex128

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
	return NewNdarrayFromSlice_uint8(d, shape)
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
	arr := python.PyArray_SimpleNew(shape, int(NPY_INT))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_int(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayFromSlice_uint(d []uint, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	arr := python.PyArray_SimpleNew(shape, int(NPY_UINT))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_uint(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayFromSlice_int8(d []int8, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	arr := python.PyArray_SimpleNew(shape, int(NPY_INT8))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_int8(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayFromSlice_uint8(d []uint8, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	arr := python.PyArray_SimpleNew(shape, int(NPY_UINT8))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_uint8(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayFromSlice_int16(d []int16, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	arr := python.PyArray_SimpleNew(shape, int(NPY_INT16))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_int16(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayFromSlice_uint16(d []uint16, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	arr := python.PyArray_SimpleNew(shape, int(NPY_UINT16))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_uint16(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayFromSlice_int32(d []int32, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	arr := python.PyArray_SimpleNew(shape, int(NPY_INT32))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_int32(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayFromSlice_uint32(d []uint32, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	arr := python.PyArray_SimpleNew(shape, int(NPY_UINT32))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_uint32(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayFromSlice_int64(d []int64, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	arr := python.PyArray_SimpleNew(shape, int(NPY_INT64))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_int64(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayFromSlice_uint64(d []uint64, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	arr := python.PyArray_SimpleNew(shape, int(NPY_UINT64))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_uint64(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayFromSlice_float32(d []float32, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	arr := python.PyArray_SimpleNew(shape, int(NPY_FLOAT32))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_float32(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

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
	arr := python.PyArray_SimpleNew(shape, int(NPY_FLOAT64))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_float64(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayFromSlice_complex64(d []complex64, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	arr := python.PyArray_SimpleNew(shape, int(NPY_COMPLEX64))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_complex64(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

	return ndarr
}

func NewNdarrayFromSlice_complex128(d []complex128, shape []int) Ndarray {
	if shape == nil {
		shape = []int{len(d)}
	}

	// Check slice length
	tot := ndarrayCheckSliceLength(len(d), shape)
	if tot < 0 {
		return nil
	}

	// Create an empty ndarray
	arr := python.PyArray_SimpleNew(shape, int(NPY_COMPLEX128))
	if arr == nil {
		return nil
	}

	// Set values
	python.PyArray_SetItems_complex128(arr, d)

	ndarr := &ndarray{arr: arr}
	setNdarrayFinalizer(ndarr)

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

func (arr *ndarray) Flags() int {
	return python.PyArray_Flags(arr.arr)
}

func (arr *ndarray) IsFortran() bool {
	return python.PyArray_IsFortran(arr.arr)
}

func (arr *ndarray) AsSlice_byte() []byte {
	return python.PyArray_GetItems_uint8(arr.arr)
}

func (arr *ndarray) AsSlice_int() []int {
	return python.PyArray_GetItems_int(arr.arr)
}

func (arr *ndarray) AsSlice_uint() []uint {
	return python.PyArray_GetItems_uint(arr.arr)
}

func (arr *ndarray) AsSlice_int8() []int8 {
	return python.PyArray_GetItems_int8(arr.arr)
}

func (arr *ndarray) AsSlice_uint8() []uint8 {
	return python.PyArray_GetItems_uint8(arr.arr)
}

func (arr *ndarray) AsSlice_int16() []int16 {
	return python.PyArray_GetItems_int16(arr.arr)
}

func (arr *ndarray) AsSlice_uint16() []uint16 {
	return python.PyArray_GetItems_uint16(arr.arr)
}

func (arr *ndarray) AsSlice_int32() []int32 {
	return python.PyArray_GetItems_int32(arr.arr)
}

func (arr *ndarray) AsSlice_uint32() []uint32 {
	return python.PyArray_GetItems_uint32(arr.arr)
}

func (arr *ndarray) AsSlice_int64() []int64 {
	return python.PyArray_GetItems_int64(arr.arr)
}

func (arr *ndarray) AsSlice_uint64() []uint64 {
	return python.PyArray_GetItems_uint64(arr.arr)
}

func (arr *ndarray) AsSlice_complex64() []complex64 {
	return python.PyArray_GetItems_complex64(arr.arr)
}

func (arr *ndarray) AsSlice_complex128() []complex128 {
	return python.PyArray_GetItems_complex128(arr.arr)
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
