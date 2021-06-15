package python

// #include "py.h"
import "C"
import (
	"fmt"
	"unsafe"
)

// NOTE: the hack method to convert C pointer to slice https://stackoverflow.com/questions/53238602/accessing-c-array-in-golang

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

// Ref: https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_DATA
func PyArray_SetItems_byte(arr *PyObject, items []byte) {
	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	dtype := PyArray_TYPE(arr)

	nitems := len(items)

	switch dtype {
	case 0: // NPY_BOOL
		panic("NPY_BOOL not supported")
	case 1: // NPY_BYTE
		dslice := (*[1 << 30]C.char)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.char(items[i])
		}

	case 2: // NPY_UBYTE
		dslice := (*[1 << 30]C.uchar)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.uchar(items[i])
		}

	case 3: // NPY_SHORT
		dslice := (*[1 << 30]C.short)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.short(items[i])
		}

	case 4: // NPY_USHORT
		dslice := (*[1 << 30]C.ushort)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.ushort(items[i])
		}

	case 5: // NPY_INT
		dslice := (*[1 << 30]C.int)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.int(items[i])
		}

	case 6: // NPY_UINT
		dslice := (*[1 << 30]C.uint)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.uint(items[i])
		}

	case 7: // NPY_LONG
		dslice := (*[1 << 30]C.long)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.long(items[i])
		}

	case 8: //NPY_ULONG
		dslice := (*[1 << 30]C.ulong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.ulong(items[i])
		}

	case 9: //NPY_LONGLONG
		dslice := (*[1 << 30]C.longlong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.longlong(items[i])
		}

	case 10: //NPY_ULONGLONG
		dslice := (*[1 << 30]C.ulonglong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.ulonglong(items[i])
		}

	case 11: //NPY_FLOAT
		dslice := (*[1 << 30]C.float)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.float(items[i])
		}

	case 12: //NPY_DOUBLE
		dslice := (*[1 << 30]C.double)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.double(items[i])
		}

	default:
		panic(fmt.Sprintf("NpyType %v not supported", dtype))
	}
}

func PyArray_SetItems_int(arr *PyObject, items []int) {
	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	dtype := PyArray_TYPE(arr)

	nitems := len(items)

	switch dtype {
	case 0: // NPY_BOOL
		panic("NPY_BOOL not supported")
	case 1: // NPY_BYTE
		dslice := (*[1 << 30]C.char)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.char(items[i])
		}

	case 2: // NPY_UBYTE
		dslice := (*[1 << 30]C.uchar)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.uchar(items[i])
		}

	case 3: // NPY_SHORT
		dslice := (*[1 << 30]C.short)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.short(items[i])
		}

	case 4: // NPY_USHORT
		dslice := (*[1 << 30]C.ushort)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.ushort(items[i])
		}

	case 5: // NPY_INT
		dslice := (*[1 << 30]C.int)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.int(items[i])
		}

	case 6: // NPY_UINT
		dslice := (*[1 << 30]C.uint)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.uint(items[i])
		}

	case 7: // NPY_LONG
		dslice := (*[1 << 30]C.long)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.long(items[i])
		}

	case 8: //NPY_ULONG
		dslice := (*[1 << 30]C.ulong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.ulong(items[i])
		}

	case 9: //NPY_LONGLONG
		dslice := (*[1 << 30]C.longlong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.longlong(items[i])
		}

	case 10: //NPY_ULONGLONG
		dslice := (*[1 << 30]C.ulonglong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.ulonglong(items[i])
		}

	case 11: //NPY_FLOAT
		dslice := (*[1 << 30]C.float)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.float(items[i])
		}

	case 12: //NPY_DOUBLE
		dslice := (*[1 << 30]C.double)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.double(items[i])
		}

	default:
		panic(fmt.Sprintf("NpyType %v not supported", dtype))
	}
}

func PyArray_SetItems_float64(arr *PyObject, items []float64) {
	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	dtype := PyArray_TYPE(arr)

	nitems := len(items)

	switch dtype {
	case 0: // NPY_BOOL
		panic("NPY_BOOL not supported")
	case 1: // NPY_BYTE
		dslice := (*[1 << 30]C.char)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.char(items[i])
		}

	case 2: // NPY_UBYTE
		dslice := (*[1 << 30]C.uchar)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.uchar(items[i])
		}

	case 3: // NPY_SHORT
		dslice := (*[1 << 30]C.short)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.short(items[i])
		}

	case 4: // NPY_USHORT
		dslice := (*[1 << 30]C.ushort)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.ushort(items[i])
		}

	case 5: // NPY_INT
		dslice := (*[1 << 30]C.int)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			dslice[i] = C.int(items[i])
		}

	case 6: // NPY_UINT
		dslice := (*[1 << 30]C.uint)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.uint(items[i])
		}

	case 7: // NPY_LONG
		dslice := (*[1 << 30]C.long)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.long(items[i])
		}

	case 8: //NPY_ULONG
		dslice := (*[1 << 30]C.ulong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.ulong(items[i])
		}

	case 9: //NPY_LONGLONG
		dslice := (*[1 << 30]C.longlong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.longlong(items[i])
		}

	case 10: //NPY_ULONGLONG
		dslice := (*[1 << 30]C.ulonglong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.ulonglong(items[i])
		}

	case 11: //NPY_FLOAT
		dslice := (*[1 << 30]C.float)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.float(items[i])
		}

	case 12: //NPY_DOUBLE
		dslice := (*[1 << 30]C.double)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			dslice[i] = C.double(items[i])
		}

	default:
		panic(fmt.Sprintf("NpyType %v not supported", dtype))
	}
}

func PyArray_GetItems_byte(arr *PyObject) []byte {
	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	dtype := PyArray_TYPE(arr)
	nitems := PyArray_Size(arr)

	items := make([]byte, nitems)

	switch dtype {
	case 0: // NPY_BOOL
		panic("NPY_BOOL not supported")
	case 1: // NPY_BYTE
		dslice := (*[1 << 30]C.char)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = byte(dslice[i])
		}

	case 2: // NPY_UBYTE
		dslice := (*[1 << 30]C.uchar)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = byte(dslice[i])
		}

	case 3: // NPY_SHORT
		dslice := (*[1 << 30]C.short)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = byte(dslice[i])
		}

	case 4: // NPY_USHORT
		dslice := (*[1 << 30]C.ushort)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = byte(dslice[i])
		}

	case 5: // NPY_INT
		dslice := (*[1 << 30]C.int)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = byte(dslice[i])
		}

	case 6: // NPY_UINT
		dslice := (*[1 << 30]C.uint)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = byte(dslice[i])
		}

	case 7: // NPY_LONG
		dslice := (*[1 << 30]C.long)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = byte(dslice[i])
		}

	case 8: //NPY_ULONG
		dslice := (*[1 << 30]C.ulong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = byte(dslice[i])
		}

	case 9: //NPY_LONGLONG
		dslice := (*[1 << 30]C.longlong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = byte(dslice[i])
		}

	case 10: //NPY_ULONGLONG
		dslice := (*[1 << 30]C.ulonglong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = byte(dslice[i])
		}

	case 11: //NPY_FLOAT
		dslice := (*[1 << 30]C.float)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = byte(dslice[i])
		}

	case 12: //NPY_DOUBLE
		dslice := (*[1 << 30]C.double)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = byte(dslice[i])
		}

	default:
		panic(fmt.Sprintf("NpyType %v not supported", dtype))
	}

	return items
}

func PyArray_GetItems_int(arr *PyObject) []int {
	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	dtype := PyArray_TYPE(arr)
	nitems := PyArray_Size(arr)

	items := make([]int, nitems)

	switch dtype {
	case 0: // NPY_BOOL
		panic("NPY_BOOL not supported")
	case 1: // NPY_BYTE
		dslice := (*[1 << 30]C.char)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = int(dslice[i])
		}

	case 2: // NPY_UBYTE
		dslice := (*[1 << 30]C.uchar)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = int(dslice[i])
		}

	case 3: // NPY_SHORT
		dslice := (*[1 << 30]C.short)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = int(dslice[i])
		}

	case 4: // NPY_USHORT
		dslice := (*[1 << 30]C.ushort)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = int(dslice[i])
		}

	case 5: // NPY_INT
		dslice := (*[1 << 30]C.int)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = int(dslice[i])
		}

	case 6: // NPY_UINT
		dslice := (*[1 << 30]C.uint)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = int(dslice[i])
		}

	case 7: // NPY_LONG
		dslice := (*[1 << 30]C.long)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = int(dslice[i])
		}

	case 8: //NPY_ULONG
		dslice := (*[1 << 30]C.ulong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = int(dslice[i])
		}

	case 9: //NPY_LONGLONG
		dslice := (*[1 << 30]C.longlong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = int(dslice[i])
		}

	case 10: //NPY_ULONGLONG
		dslice := (*[1 << 30]C.ulonglong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = int(dslice[i])
		}

	case 11: //NPY_FLOAT
		dslice := (*[1 << 30]C.float)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = int(dslice[i])
		}

	case 12: //NPY_DOUBLE
		dslice := (*[1 << 30]C.double)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = int(dslice[i])
		}

	default:
		panic(fmt.Sprintf("NpyType %v not supported", dtype))
	}

	return items
}

func PyArray_GetItems_float64(arr *PyObject) []float64 {
	dptr := unsafe.Pointer(C.__PyArray_DATA(go2c(arr)))
	dtype := PyArray_TYPE(arr)
	nitems := PyArray_Size(arr)

	items := make([]float64, nitems)

	switch dtype {
	case 0: // NPY_BOOL
		panic("NPY_BOOL not supported")
	case 1: // NPY_BYTE
		dslice := (*[1 << 30]C.char)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = float64(dslice[i])
		}

	case 2: // NPY_UBYTE
		dslice := (*[1 << 30]C.uchar)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = float64(dslice[i])
		}

	case 3: // NPY_SHORT
		dslice := (*[1 << 30]C.short)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = float64(dslice[i])
		}

	case 4: // NPY_USHORT
		dslice := (*[1 << 30]C.ushort)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = float64(dslice[i])
		}

	case 5: // NPY_INT
		dslice := (*[1 << 30]C.int)(dptr)[:nitems:nitems]
		for i := 0; i < nitems; i++ {
			items[i] = float64(dslice[i])
		}

	case 6: // NPY_UINT
		dslice := (*[1 << 30]C.uint)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = float64(dslice[i])
		}

	case 7: // NPY_LONG
		dslice := (*[1 << 30]C.long)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = float64(dslice[i])
		}

	case 8: //NPY_ULONG
		dslice := (*[1 << 30]C.ulong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = float64(dslice[i])
		}

	case 9: //NPY_LONGLONG
		dslice := (*[1 << 30]C.longlong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = float64(dslice[i])
		}

	case 10: //NPY_ULONGLONG
		dslice := (*[1 << 30]C.ulonglong)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = float64(dslice[i])
		}

	case 11: //NPY_FLOAT
		dslice := (*[1 << 30]C.float)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = float64(dslice[i])
		}

	case 12: //NPY_DOUBLE
		dslice := (*[1 << 30]C.double)(dptr)[:len(items):len(items)]
		for i := 0; i < len(items); i++ {
			items[i] = float64(dslice[i])
		}

	default:
		panic(fmt.Sprintf("NpyType %v not supported", dtype))
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
