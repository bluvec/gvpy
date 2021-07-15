package gvpy

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"unsafe"

	"github.com/bluvec/gvpy/python"
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
	PyObjectGetter

	Ndim() int
	Shape() []int
	Dtype() NpyType
	Size() int
	Flags() int
	IsFortran() bool
	DeepCopy() Ndarray // deep copy
	Close()

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

	// low-level API
	NdimX() int
	ShapeX() []int
	DtypeX() NpyType
	SizeX() int
	FlagsX() int
	IsFortranX() bool
	DeepCopyX() Ndarray // deep copy
	CloseX()

	AsSliceX_byte() []byte
	AsSliceX_int() []int
	AsSliceX_uint() []uint
	AsSliceX_int8() []int8
	AsSliceX_uint8() []uint8
	AsSliceX_int16() []int16
	AsSliceX_uint16() []uint16
	AsSliceX_int32() []int32
	AsSliceX_uint32() []uint32
	AsSliceX_int64() []int64
	AsSliceX_uint64() []uint64
	AsSliceX_complex64() []complex64
	AsSliceX_complex128() []complex128
}

type ndarray struct {
	arr *python.PyObject
}

// constructors
func NewNdarray(shape []int, dtype NpyType) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarray(shape, dtype)
}

func NewNdarrayZeros(shape []int, dtype NpyType, fortran bool) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayZeros(shape, dtype, fortran)
}

func NewNdarrayEmpty(shape []int, dtype NpyType, fortran bool) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayEmpty(shape, dtype, fortran)
}

func NewNdarrayFromPyObject(arr *python.PyObject) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromPyObject(arr)
}

func NewNdarrayFromSlice_byte(d []byte, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_byte(d, shape)
}

func NewNdarrayFromSlice_int(d []int, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_int(d, shape)
}

func NewNdarrayFromSlice_uint(d []uint, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_uint(d, shape)
}

func NewNdarrayFromSlice_int8(d []int8, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_int8(d, shape)
}

func NewNdarrayFromSlice_uint8(d []uint8, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_uint8(d, shape)
}

func NewNdarrayFromSlice_int16(d []int16, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_int16(d, shape)
}

func NewNdarrayFromSlice_uint16(d []uint16, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_uint16(d, shape)
}

func NewNdarrayFromSlice_int32(d []int32, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_int32(d, shape)
}

func NewNdarrayFromSlice_uint32(d []uint32, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_uint32(d, shape)
}

func NewNdarrayFromSlice_int64(d []int64, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_int64(d, shape)
}

func NewNdarrayFromSlice_uint64(d []uint64, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_uint64(d, shape)
}

func NewNdarrayFromSlice_float32(d []float32, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_float32(d, shape)
}

func NewNdarrayFromSlice_float64(d []float64, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_float64(d, shape)
}

func NewNdarrayFromSlice_complex64(d []complex64, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_complex64(d, shape)
}

func NewNdarrayFromSlice_complex128(d []complex128, shape []int) Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return newNdarrayFromSlice_complex128(d, shape)
}

// methods
func (arr *ndarray) String() string {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.arr.String()
}

func (arr *ndarray) PyObject() *python.PyObject {
	return arr.arr
}

func (arr *ndarray) Ndim() int {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.ndim()
}

func (arr *ndarray) Shape() []int {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.shape()
}

func (arr *ndarray) Dtype() NpyType {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.dtype()
}

func (arr *ndarray) Size() int {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.size()
}

func (arr *ndarray) Flags() int {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.flags()
}

func (arr *ndarray) IsFortran() bool {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.isFortran()
}

func (arr *ndarray) DeepCopy() Ndarray {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.deepCopy()
}

func (arr *ndarray) Close() {
	gil := GILEnsure()
	defer GILRelease(gil)

	arr.close()
}

func (arr *ndarray) AsSlice_byte() []byte {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.asSlice_byte()
}

func (arr *ndarray) AsSlice_int() []int {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.asSlice_int()
}

func (arr *ndarray) AsSlice_uint() []uint {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.asSlice_uint()
}

func (arr *ndarray) AsSlice_int8() []int8 {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.asSlice_int8()
}

func (arr *ndarray) AsSlice_uint8() []uint8 {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.asSlice_uint8()
}

func (arr *ndarray) AsSlice_int16() []int16 {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.asSlice_int16()
}

func (arr *ndarray) AsSlice_uint16() []uint16 {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.asSlice_uint16()
}

func (arr *ndarray) AsSlice_int32() []int32 {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.asSlice_int32()
}

func (arr *ndarray) AsSlice_uint32() []uint32 {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.asSlice_uint32()
}

func (arr *ndarray) AsSlice_int64() []int64 {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.asSlice_int64()
}

func (arr *ndarray) AsSlice_uint64() []uint64 {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.asSlice_uint64()
}

func (arr *ndarray) AsSlice_complex64() []complex64 {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.asSlice_complex64()
}

func (arr *ndarray) AsSlice_complex128() []complex128 {
	gil := GILEnsure()
	defer GILRelease(gil)

	return arr.asSlice_complex128()
}

///////////////////////////////////////////////////////////////
// Lowlevel APIs
///////////////////////////////////////////////////////////////
// constructors
func NewNdarrayX(shape []int, dtype NpyType) Ndarray {
	return newNdarray(shape, dtype)
}

func NewNdarrayZerosX(shape []int, dtype NpyType, fortran bool) Ndarray {
	return newNdarrayZeros(shape, dtype, fortran)
}

func NewNdarrayEmptyX(shape []int, dtype NpyType, fortran bool) Ndarray {
	return newNdarrayEmpty(shape, dtype, fortran)
}

func NewNdarrayFromPyObjectX(arr *python.PyObject) Ndarray {
	return newNdarrayFromPyObject(arr)
}

func NewNdarrayFromSliceX_byte(d []byte, shape []int) Ndarray {
	return newNdarrayFromSlice_byte(d, shape)
}

func NewNdarrayFromSliceX_int(d []int, shape []int) Ndarray {
	return newNdarrayFromSlice_int(d, shape)
}

func NewNdarrayFromSliceX_uint(d []uint, shape []int) Ndarray {
	return newNdarrayFromSlice_uint(d, shape)
}

func NewNdarrayFromSliceX_int8(d []int8, shape []int) Ndarray {
	return newNdarrayFromSlice_int8(d, shape)
}

func NewNdarrayFromSliceX_uint8(d []uint8, shape []int) Ndarray {
	return newNdarrayFromSlice_uint8(d, shape)
}

func NewNdarrayFromSliceX_int16(d []int16, shape []int) Ndarray {
	return newNdarrayFromSlice_int16(d, shape)
}

func NewNdarrayFromSliceX_uint16(d []uint16, shape []int) Ndarray {
	return newNdarrayFromSlice_uint16(d, shape)
}

func NewNdarrayFromSliceX_int32(d []int32, shape []int) Ndarray {
	return newNdarrayFromSlice_int32(d, shape)
}

func NewNdarrayFromSliceX_uint32(d []uint32, shape []int) Ndarray {
	return newNdarrayFromSlice_uint32(d, shape)
}

func NewNdarrayFromSliceX_int64(d []int64, shape []int) Ndarray {
	return newNdarrayFromSlice_int64(d, shape)
}

func NewNdarrayFromSliceX_uint64(d []uint64, shape []int) Ndarray {
	return newNdarrayFromSlice_uint64(d, shape)
}

func NewNdarrayFromSliceX_float32(d []float32, shape []int) Ndarray {
	return newNdarrayFromSlice_float32(d, shape)
}

func NewNdarrayFromSliceX_float64(d []float64, shape []int) Ndarray {
	return newNdarrayFromSlice_float64(d, shape)
}

func NewNdarrayFromSliceX_complex64(d []complex64, shape []int) Ndarray {
	return newNdarrayFromSlice_complex64(d, shape)
}

func NewNdarrayFromSliceX_complex128(d []complex128, shape []int) Ndarray {
	return newNdarrayFromSlice_complex128(d, shape)
}

// methods
func (arr *ndarray) NdimX() int {
	return arr.ndim()
}

func (arr *ndarray) ShapeX() []int {
	return arr.shape()
}

func (arr *ndarray) DtypeX() NpyType {
	return arr.dtype()
}

func (arr *ndarray) SizeX() int {
	return arr.size()
}

func (arr *ndarray) FlagsX() int {
	return arr.flags()
}

func (arr *ndarray) IsFortranX() bool {
	return arr.isFortran()
}

func (arr *ndarray) DeepCopyX() Ndarray {
	return arr.deepCopy()
}

func (arr *ndarray) CloseX() {
	arr.close()
}

func (arr *ndarray) AsSliceX_byte() []byte {
	return arr.asSlice_byte()
}

func (arr *ndarray) AsSliceX_int() []int {
	return arr.asSlice_int()
}

func (arr *ndarray) AsSliceX_uint() []uint {
	return arr.asSlice_uint()
}

func (arr *ndarray) AsSliceX_int8() []int8 {
	return arr.asSlice_int8()
}

func (arr *ndarray) AsSliceX_uint8() []uint8 {
	return arr.asSlice_uint8()
}

func (arr *ndarray) AsSliceX_int16() []int16 {
	return arr.asSlice_int16()
}

func (arr *ndarray) AsSliceX_uint16() []uint16 {
	return arr.asSlice_uint16()
}

func (arr *ndarray) AsSliceX_int32() []int32 {
	return arr.asSlice_int32()
}

func (arr *ndarray) AsSliceX_uint32() []uint32 {
	return arr.asSlice_uint32()
}

func (arr *ndarray) AsSliceX_int64() []int64 {
	return arr.asSlice_int64()
}

func (arr *ndarray) AsSliceX_uint64() []uint64 {
	return arr.asSlice_uint64()
}

func (arr *ndarray) AsSliceX_complex64() []complex64 {
	return arr.asSlice_complex64()
}

func (arr *ndarray) AsSliceX_complex128() []complex128 {
	return arr.asSlice_complex128()
}

///////////////////////////////////////////////////////////////
// Private functions
///////////////////////////////////////////////////////////////
func (arr *ndarray) setFinalizer() {
	if gcEnabled {
		runtime.SetFinalizer(arr, func(arr *ndarray) { arr.close() })
	}
}

func (arr *ndarray) close() {
	pyobj := atomic.SwapPointer((*unsafe.Pointer)(unsafe.Pointer(&arr.arr)), nil)
	if pyobj != nil {
		python.Py_CLEAR((*python.PyObject)(pyobj))
	}
}

// constructors
func newNdarray(shape []int, dtype NpyType) *ndarray {
	arr := python.PyArray_SimpleNew(shape, int(dtype))
	if arr == nil {
		return nil
	}

	ndarr := &ndarray{arr: arr}
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayZeros(shape []int, dtype NpyType, fortran bool) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayEmpty(shape []int, dtype NpyType, fortran bool) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromPyObject(arr *python.PyObject) *ndarray {
	if arr == nil {
		return nil
	}

	python.Py_XINCREF(arr)
	ndarr := &ndarray{arr: arr}
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_byte(d []byte, shape []int) *ndarray {
	return newNdarrayFromSlice_uint8(d, shape)
}

func newNdarrayFromSlice_int(d []int, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_uint(d []uint, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_int8(d []int8, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_uint8(d []uint8, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_int16(d []int16, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_uint16(d []uint16, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_int32(d []int32, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_uint32(d []uint32, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_int64(d []int64, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_uint64(d []uint64, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_float32(d []float32, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_float64(d []float64, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_complex64(d []complex64, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

func newNdarrayFromSlice_complex128(d []complex128, shape []int) *ndarray {
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
	ndarr.setFinalizer()

	return ndarr
}

// methods
func (arr *ndarray) ndim() int {
	return python.PyArray_NDIM(arr.arr)
}

func (arr *ndarray) shape() []int {
	return python.PyArray_Shape(arr.arr)
}

func (arr *ndarray) dtype() NpyType {
	return NpyType(python.PyArray_TYPE(arr.arr))
}

func (arr *ndarray) size() int {
	return python.PyArray_Size(arr.arr)
}

func (arr *ndarray) flags() int {
	return python.PyArray_Flags(arr.arr)
}

func (arr *ndarray) isFortran() bool {
	return python.PyArray_IsFortran(arr.arr)
}

func (arr *ndarray) asSlice_byte() []byte {
	return python.PyArray_GetItems_uint8(arr.arr)
}

func (arr *ndarray) asSlice_int() []int {
	return python.PyArray_GetItems_int(arr.arr)
}

func (arr *ndarray) asSlice_uint() []uint {
	return python.PyArray_GetItems_uint(arr.arr)
}

func (arr *ndarray) asSlice_int8() []int8 {
	return python.PyArray_GetItems_int8(arr.arr)
}

func (arr *ndarray) asSlice_uint8() []uint8 {
	return python.PyArray_GetItems_uint8(arr.arr)
}

func (arr *ndarray) asSlice_int16() []int16 {
	return python.PyArray_GetItems_int16(arr.arr)
}

func (arr *ndarray) asSlice_uint16() []uint16 {
	return python.PyArray_GetItems_uint16(arr.arr)
}

func (arr *ndarray) asSlice_int32() []int32 {
	return python.PyArray_GetItems_int32(arr.arr)
}

func (arr *ndarray) asSlice_uint32() []uint32 {
	return python.PyArray_GetItems_uint32(arr.arr)
}

func (arr *ndarray) asSlice_int64() []int64 {
	return python.PyArray_GetItems_int64(arr.arr)
}

func (arr *ndarray) asSlice_uint64() []uint64 {
	return python.PyArray_GetItems_uint64(arr.arr)
}

func (arr *ndarray) asSlice_complex64() []complex64 {
	return python.PyArray_GetItems_complex64(arr.arr)
}

func (arr *ndarray) asSlice_complex128() []complex128 {
	return python.PyArray_GetItems_complex128(arr.arr)
}

func (arr *ndarray) deepCopy() *ndarray {
	newarr := newNdarray(arr.shape(), arr.dtype())
	if newarr == nil {
		return nil
	}

	if err := python.PyArray_Copy(arr.arr, newarr.arr); err != nil {
		return nil
	}

	return newarr
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
