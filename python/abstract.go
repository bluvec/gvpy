package python

// #include "py.h"
import "C"

// Ref: https://docs.python.org/3.8/c-api/object.html?#c.PyObject_IsInstance
func PyObject_IsInstance(inst *PyObject, class *PyObject) bool {
	return int(C.__PyObject_IsInstance(go2c(inst), go2c(class))) == 1
}
