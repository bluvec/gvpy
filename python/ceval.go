package python

// #include "py.h"
import "C"

type PyThreadState struct {
	p *C.PyThreadState
}

func PyEval_SaveThread() PyThreadState {
	p := C.__PyEval_SaveThread()
	tstate := PyThreadState{p: p}
	return tstate
}

func PyEval_RestoreThread(tstate PyThreadState) {
	C.__PyEval_RestoreThread(tstate.p)
}
