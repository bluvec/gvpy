package gvpy

import (
	"fmt"
	"gvpy/python"
	"strconv"
	"strings"
)

type ThreadState python.PyThreadState
type GILState python.PyGILState_STATE

func Initialize() error {
	python.Py_Initialize()

	if !python.Py_IsInitialized() {
		return fmt.Errorf("gvpy: error to initialize the python interpreter")
	}

	// Initialize numpy module and PyArray functions
	return python.PyArray_import_array()
}

// Ref: https://docs.python.org/3/c-api/init.html?#c.PyEval_InitThreads
// make sure the GIL is correctly initialized: python < 3.7
// Call this function to initialize GIL if your python version is less than 3.7.
func InitThreads() error {
	python.PyEval_InitThreads()
	if !python.PyEval_ThreadsInitialized() {
		return fmt.Errorf("gvpy: error to initialize the python GIL")
	}
	return nil
}

func Finalize() {
	python.Py_FinalizeEx()
}

func SaveThread() ThreadState {
	return ThreadState(python.PyEval_SaveThread())
}

func RestoreThread(s ThreadState) {
	python.PyEval_RestoreThread((python.PyThreadState(s)))
}

func GILStateCheck() bool {
	return python.PyGILState_Check()
}

func GILEnsure() GILState {
	return GILState(python.PyGILState_Ensure())
}

func GILRelease(s GILState) {
	python.PyGILState_Release(python.PyGILState_STATE(s))
}

func PyErrPrint() {
	python.PyErr_Print()
}

func GetVersion() string {
	return python.Py_GetVersion()
}

func GetPyVersion() []int {
	verStr := GetVersion()
	pyVerStr := verStr[0:strings.Index(verStr, " ")]
	pyVerStrList := strings.Split(pyVerStr, ".")

	var err error
	pyVerIntList := make([]int, len(pyVerStrList))

	for i := range pyVerStrList {
		if pyVerIntList[i], err = strconv.Atoi(pyVerStrList[i]); err != nil {
			return nil
		}
	}

	return pyVerIntList
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

func GetSysPath() []string {
	sysPathObj := python.PySys_GetObject("path")
	nPath := python.PyList_Size(sysPathObj)
	sysPath := make([]string, nPath)

	for i := 0; i < nPath; i++ {
		o := python.PyList_GetItem(sysPathObj, i)
		sysPath[i] = python.PyUnicode_AsString(o)
	}

	return sysPath
}

func SetSysPath(path []string) {
	python.PySys_SetPath(strings.Join(path, python.PyDelimiter))
}

func AddSysPath(path string) {
	sysPathObj := python.PySys_GetObject("path")
	pathObj := python.PyUnicode_FromString(path)
	python.PyList_Append(sysPathObj, pathObj)
}

func AddSysPathAtFront(path string) {
	sysPathObj := python.PySys_GetObject("path")
	pathObj := python.PyUnicode_FromString(path)
	python.PyList_Insert(sysPathObj, 0, pathObj)
}
