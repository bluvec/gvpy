package python

// #include "py.h"
import "C"

type PyGILState_STATE struct {
	s C.__PyGILState_STATE
}

// Ref: https://docs.python.org/3.8/c-api/init.html?#c.PyGILState_Check
func PyGILState_Check() bool {
	if int(C.__PyGILState_Check()) == 1 {
		return true
	} else {
		return false
	}
}

// Ref: https://docs.python.org/3.8/c-api/init.html?#c.PyGILState_Ensure
func PyGILState_Ensure() PyGILState_STATE {
	s := C.__PyGILState_Ensure()
	return PyGILState_STATE{s: s}
}

// Ref: https://docs.python.org/3.8/c-api/init.html?#c.PyGILState_Release
func PyGILState_Release(s PyGILState_STATE) {
	C.__PyGILState_Release(s.s)
}

func (gil PyGILState_STATE) String() string {
	if gil.s == C.int(0) {
		return "PyGILState_LOCKED"
	} else {
		return "PyGILState_UNLOCKED"
	}

}
