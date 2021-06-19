package python

// #include "py.h"
import "C"
import (
	"fmt"
	"unsafe"
)

func PyRun_SimpleString(cmd string) error {
	cCmd := C.CString(cmd)
	defer C.free(unsafe.Pointer(cCmd))

	rc := int(C.__PyRun_SimpleString(cCmd))
	if rc != 0 {
		return fmt.Errorf("error to call PyRun_SimpleString(%v)", cmd)
	}

	return nil
}
