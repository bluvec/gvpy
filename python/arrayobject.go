package python

// #include "py.h"
import "C"
import (
	"fmt"
	"unsafe"
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

// Ref: https://numpy.org/doc/stable/reference/c-api/array.html?#c.import_array
// Ref: https://numpy.org/doc/stable/user/c-info.how-to-extend.html?#how-to-extend-numpy
func PyArray_import_array() error {
	if int(C.__PyArray_import_array()) != 0 {
		return fmt.Errorf("numpy.core.multiarray failed to import")
	}
	return nil
}

// Ref: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_Check
func PyArray_Check(o *PyObject) bool {
	return int(C.__PyArray_Check(go2c(o))) == 1
}

// int PyArray_NDIM(PyArrayObject *arr)
// Ref: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_NDIM
func PyArray_NDIM(arr *PyObject) int {
	return int(C.__PyArray_NDIM(go2c(arr)))
}

// int PyArray_FLAGS(PyArrayObject* arr)
// Ref: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_FLAGS
func PyArray_FLAGS(arr *PyObject) int {
	return int(C.__PyArray_FLAGS(go2c(arr)))
}

// int PyArray_TYPE(PyArrayObject* arr)
// Ref: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_TYPE
func PyArray_TYPE(arr *PyObject) int {
	return int(C.__PyArray_TYPE(go2c(arr)))
}

func PyArray_Shape(arr *PyObject) []int {
	ndim := int(C.__PyArray_NDIM(go2c(arr)))
	dptr := unsafe.Pointer(C.__PyArray_DIMS(go2c(arr)))
	dslice := (*[1 << 30]C.npy_intp)(dptr)[:ndim:ndim]

	shape := make([]int, ndim)
	for i := 0; i < ndim; i++ {
		shape[i] = int(dslice[i])
	}

	return shape
}

// Ref: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_Size
func PyArray_Size(arr *PyObject) int {
	return int(C.__PyArray_Size(go2c(arr)))
}

// NOTE: the hack method to convert C pointer to slice https://stackoverflow.com/questions/53238602/accessing-c-array-in-golang
// Ref: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_DATA
func PyArray_SetItems_uint(arr *PyObject, items []uint) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_UINT) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.uint)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.uint(items[i])
	}
}

func PyArray_SetItems_int(arr *PyObject, items []int) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_INT) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.int)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.int(items[i])
	}
}

func PyArray_SetItems_uint8(arr *PyObject, items []uint8) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_UINT8) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.uint8_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.uint8_t(items[i])
	}
}

func PyArray_SetItems_int8(arr *PyObject, items []int8) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_INT8) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.int8_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.int8_t(items[i])
	}
}

func PyArray_SetItems_uint16(arr *PyObject, items []uint16) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_UINT16) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.uint16_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.uint16_t(items[i])
	}
}

func PyArray_SetItems_int16(arr *PyObject, items []int16) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_INT16) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.int16_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.int16_t(items[i])
	}
}

func PyArray_SetItems_uint32(arr *PyObject, items []uint32) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_UINT32) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.uint32_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.uint32_t(items[i])
	}
}

func PyArray_SetItems_int32(arr *PyObject, items []int32) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_INT32) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.int32_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.int32_t(items[i])
	}
}

func PyArray_SetItems_uint64(arr *PyObject, items []uint64) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_UINT64) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.uint64_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.uint64_t(items[i])
	}
}

func PyArray_SetItems_int64(arr *PyObject, items []int64) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_INT64) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.int64_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.int64_t(items[i])
	}
}

func PyArray_SetItems_float32(arr *PyObject, items []float32) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_FLOAT32) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.float)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.float(items[i])
	}
}

func PyArray_SetItems_float64(arr *PyObject, items []float64) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_FLOAT64) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.double)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.double(items[i])
	}
}

func PyArray_SetItems_complex64(arr *PyObject, items []complex64) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_COMPLEX64) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.complexfloat)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.complexfloat(items[i])
	}
}

func PyArray_SetItems_complex128(arr *PyObject, items []complex128) {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_COMPLEX128) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := len(items)
	dslice := (*[1 << 30]C.complexdouble)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		dslice[i] = C.complexdouble(items[i])
	}
}

func PyArray_GetItems_uint(arr *PyObject) []uint {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_UINT) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]uint, nitems)
	dslice := (*[1 << 30]C.uint)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = uint(dslice[i])
	}

	return items
}

func PyArray_GetItems_int(arr *PyObject) []int {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_INT) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]int, nitems)
	dslice := (*[1 << 30]C.int)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = int(dslice[i])
	}

	return items
}

func PyArray_GetItems_uint8(arr *PyObject) []uint8 {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_UINT8) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]uint8, nitems)
	dslice := (*[1 << 30]C.uint8_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = uint8(dslice[i])
	}

	return items
}

func PyArray_GetItems_int8(arr *PyObject) []int8 {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_INT8) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]int8, nitems)
	dslice := (*[1 << 30]C.int8_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = int8(dslice[i])
	}

	return items
}

func PyArray_GetItems_uint16(arr *PyObject) []uint16 {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_UINT16) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]uint16, nitems)
	dslice := (*[1 << 30]C.uint16_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = uint16(dslice[i])
	}

	return items
}

func PyArray_GetItems_int16(arr *PyObject) []int16 {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_INT16) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]int16, nitems)
	dslice := (*[1 << 30]C.int16_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = int16(dslice[i])
	}

	return items
}

func PyArray_GetItems_uint32(arr *PyObject) []uint32 {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_UINT32) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]uint32, nitems)
	dslice := (*[1 << 30]C.uint32_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = uint32(dslice[i])
	}

	return items
}

func PyArray_GetItems_int32(arr *PyObject) []int32 {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_INT32) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]int32, nitems)
	dslice := (*[1 << 30]C.int32_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = int32(dslice[i])
	}

	return items
}

func PyArray_GetItems_uint64(arr *PyObject) []uint64 {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_UINT64) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]uint64, nitems)
	dslice := (*[1 << 30]C.uint64_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = uint64(dslice[i])
	}

	return items
}

func PyArray_GetItems_int64(arr *PyObject) []int64 {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_INT64) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]int64, nitems)
	dslice := (*[1 << 30]C.int64_t)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = int64(dslice[i])
	}

	return items
}

func PyArray_GetItems_float32(arr *PyObject) []float32 {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_FLOAT32) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]float32, nitems)
	dslice := (*[1 << 30]C.float)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = float32(dslice[i])
	}

	return items
}

func PyArray_GetItems_float64(arr *PyObject) []float64 {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_FLOAT64) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]float64, nitems)
	dslice := (*[1 << 30]C.double)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = float64(dslice[i])
	}

	return items
}

func PyArray_GetItems_complex64(arr *PyObject) []complex64 {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_COMPLEX64) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]complex64, nitems)
	dslice := (*[1 << 30]C.complexfloat)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = complex64(dslice[i])
	}

	return items
}

func PyArray_GetItems_complex128(arr *PyObject) []complex128 {
	dtype := PyArray_TYPE(arr)
	if dtype != int(NPY_COMPLEX128) {
		panic("PyArray dtype mismatch")
	}

	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	nitems := PyArray_Size(arr)
	items := make([]complex128, nitems)
	dslice := (*[1 << 30]C.complexdouble)(dptr)[:nitems:nitems]
	for i := 0; i < nitems; i++ {
		items[i] = complex128(dslice[i])
	}

	return items
}

// PyObject *PyArray_SimpleNew(int nd, npy_intp const *dims, int typenum)
// Ref: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_SimpleNew
func PyArray_SimpleNew(dims []int, typenum int) *PyObject {
	cdims := make([]C.npy_intp, len(dims))
	for i := range dims {
		cdims[i] = C.npy_intp(dims[i])
	}
	return c2go(C.__PyArray_SimpleNew(C.int(len(dims)), &cdims[0], C.int(typenum)))
}

// https://numpy.org/doc/stable/reference/c-api/array.html?#c.PyArray_ZEROS
func PyArray_ZEROS(dims []int, typenum int, fortran int) *PyObject {
	nd := len(dims)
	cdims := make([]C.npy_intp, nd)
	for i := 0; i < nd; i++ {
		cdims[i] = C.npy_intp(dims[i])
	}
	return c2go(C.__PyArray_ZEROS(C.int(nd), &cdims[0], C.int(typenum), C.int(fortran)))
}

// Ref: https://numpy.org/doc/stable/reference/c-api/array.html?#c.PyArray_Empty
func PyArray_EMPTY(dims []int, typenum int, fortran int) *PyObject {
	cdims := make([]C.npy_intp, len(dims))
	for i := range dims {
		cdims[i] = C.npy_intp(dims[i])
	}
	return c2go(C.__PyArray_EMPTY(C.int(len(dims)), &cdims[0], C.int(typenum), C.int(fortran)))
}

// Ref: https://numpy.org/doc/stable/reference/c-api/array.html?#c.PyArray_INCREF
func PyArray_INCREF(arr *PyObject) error {
	if int(C.__PyArray_INCREF(go2c(arr))) != 0 {
		return fmt.Errorf("error to PyArray_INCREF")
	} else {
		return nil
	}
}

// Ref: https://numpy.org/doc/stable/reference/c-api/array.html?#c.PyArray_XDECREF
func PyArray_XDECREF(arr *PyObject) error {
	if int(C.__PyArray_XDECREF(go2c(arr))) != 0 {
		return fmt.Errorf("error to _PyArray_XDECREF")
	} else {
		return nil
	}
}

func PyArray_Copy(sarr *PyObject, darr *PyObject) error {
	ret := int(C.__PyArray_Copy(go2c(sarr), go2c(darr)))
	if ret == 0 {
		return nil
	} else {
		return fmt.Errorf("error to copy ndarray")
	}
}

func PyArray_Flags(arr *PyObject) int {
	return int(C.__PyArray_FLAGS(go2c(arr)))
}

func PyArray_IsFortran(arr *PyObject) bool {
	return C.__PyArray_IsFortran(go2c(arr)) != C.int(0)
}
