package python

// #include "py.h"
import "C"

func Py_Version() string {
	return C.GoString(C.__Py_VERSION())
}

func Py_VersionHex() int {
	return int(C.__Py_VERSION_HEX())
}

func Py_VersionMajor() int {
	return int(C.__Py_MAJOR_VERSION())
}

func Py_VersionMinor() int {
	return int(C.__Py_MINOR_VERSION())
}

func Py_VersionMicro() int {
	return int(C.__Py_MICRO_VERSION())
}
