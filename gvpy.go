package gvpy

import (
	"fmt"
	"runtime"
	"strings"

	"github.com/bluvec/gvpy/python"
)

type ThreadState python.PyThreadState
type GILState python.PyGILState_STATE

var gcEnabled bool = false
var gThreadState ThreadState

func Initialize() error {
	return Initialize2(true)
}

func Initialize2(enableGC bool) error {
	gcEnabled = enableGC

	if enableGC {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		err := InitializeX()
		if err != nil {
			return err
		}
		SaveThreadX()
		return nil
	} else {
		runtime.LockOSThread()

		err := InitializeX()
		if err != nil {
			return err
		}
		gThreadState = SaveThreadX()
		return nil
	}
}

func Finalize() {
	if gcEnabled {
		Run("import sys; sys.stdout.flush()")
	} else {
		RestoreThreadX(gThreadState)
		FinalizeX()
		runtime.UnlockOSThread()
	}
}

func (gil GILState) String() string {
	return python.PyGILState_STATE(gil).String()
}

func GILEnsure() GILState {
	runtime.LockOSThread()
	gil := GILEnsureX()
	return gil
}

func GILRelease(s GILState) {
	GILReleaseX(s)
	runtime.UnlockOSThread()
}

func GILStateCheck() bool {
	return python.PyGILState_Check()
}

func PyErrPrint() {
	gil := GILEnsure()
	defer GILRelease(gil)

	python.PyErr_Print()
}

func PyErrClear() {
	gil := GILEnsure()
	defer GILRelease(gil)

	python.PyErr_Clear()
}

func PyVersionDetail() string {
	return python.Py_GetVersion()
}

func PyVersion() string {
	return python.Py_Version()
}

func PyVersionHex() int {
	return python.Py_VersionHex()
}

func PyVersionList() []int {
	return []int{python.Py_VersionMajor(), python.Py_VersionMinor(), python.Py_VersionMicro()}
}

func GetPath() string {
	return python.Py_GetPath()
}

func GetPaths() []string {
	return strings.Split(GetPath(), python.PyDelimiter)
}

// Set before Initialize to be effective
func SetPath(path string) {
	python.Py_SetPath(path)
}

// Set before Initialize to be effective
func SetPaths(paths []string) {
	SetPath(strings.Join(paths, python.PyDelimiter))
}

// Add before Initialize to be effective
func AddPath(path string) {
	paths := GetPath()
	if strings.HasSuffix(paths, python.PyDelimiter) {
		SetPath(paths + path)
	} else {
		SetPath(paths + python.PyDelimiter + path)
	}
}

// Add before Initialize to be effective
func AddPathAtFront(path string) {
	paths := GetPath()
	if strings.HasPrefix(paths, python.PyDelimiter) {
		SetPath(path + paths)
	} else {
		SetPath(path + python.PyDelimiter + paths)
	}
}

// Get sys.path.
func GetSysPath() []string {
	gil := GILEnsure()
	defer GILRelease(gil)

	return GetSysPathX()
}

func SetSysPath(path []string) {
	gil := GILEnsure()
	defer GILRelease(gil)

	SetSysPathX(path)
}

func AddSysPath(path string) {
	gil := GILEnsure()
	defer GILRelease(gil)

	AddSysPathX(path)
}

func AddSysPathAtFront(path string) {
	gil := GILEnsure()
	defer GILRelease(gil)

	AddSysPathAtFrontX(path)
}

// Low-level API: use Initialize if possible.
func InitializeX() error {
	python.Py_InitializeEx(0)

	if !python.Py_IsInitialized() {
		return fmt.Errorf("gvpy: error to initialize the python interpreter")
	}

	// Ref: https://docs.python.org/3/c-api/init.html?#c.PyEval_InitThreads
	// make sure the GIL is correctly initialized: python < 3.7
	if python.Py_VersionHex() < 0x03070000 {
		python.PyEval_InitThreads()
		if !python.PyEval_ThreadsInitialized() {
			return fmt.Errorf("gvpy: error to initialize the python GIL")
		}
	}

	// Initialize numpy module and PyArray functions
	return python.PyArray_import_array()
}

// Low-level API: use Finalize if possible.
func FinalizeX() {
	python.Py_FinalizeEx()
}

// Low-level API: use GILEnsure if possible.
func GILEnsureX() GILState {
	return GILState(python.PyGILState_Ensure())
}

// Low-level API: use GILRelease if possible.
func GILReleaseX(s GILState) {
	python.PyGILState_Release(python.PyGILState_STATE(s))
}

// Lowlevel API: save the current thread state.
func SaveThreadX() ThreadState {
	return ThreadState(python.PyEval_SaveThread())
}

// Lowlevel API: restore the thread state.
func RestoreThreadX(s ThreadState) {
	python.PyEval_RestoreThread((python.PyThreadState(s)))
}

// Lowlevel API: print the python error message.
func PyErrPrintX() {
	python.PyErr_Print()
}

// Lowlevel API: get sys.path.
func GetSysPathX() []string {
	sysPathObj := python.PySys_GetObject("path")
	nPath := python.PyList_Size(sysPathObj)
	sysPath := make([]string, nPath)

	for i := 0; i < nPath; i++ {
		o := python.PyList_GetItem(sysPathObj, i)
		sysPath[i] = python.PyUnicode_AsString(o)
	}

	return sysPath
}

// Lowlevel API: set sys.path.
func SetSysPathX(path []string) {
	python.PySys_SetPath(strings.Join(path, python.PyDelimiter))
}

// Lowlevel API: add to sys.path.
func AddSysPathX(path string) {
	sysPathObj := python.PySys_GetObject("path")
	pathObj := python.PyUnicode_FromString(path)
	python.PyList_Append(sysPathObj, pathObj)
}

// Lowlevel API: add to the front of sys.path.
func AddSysPathAtFrontX(path string) {
	sysPathObj := python.PySys_GetObject("path")
	pathObj := python.PyUnicode_FromString(path)
	python.PyList_Insert(sysPathObj, 0, pathObj)
}
