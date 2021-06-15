package python

// #include "py.h"
import "C"

// Ref: https://wiki.python.org/moin/Py3kExtensionModules
func PyClass_Check(o *PyObject) bool {
	return int(C.__PyClass_Check(go2c(o))) == 1
}
