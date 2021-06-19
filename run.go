package gvpy

import "github.com/bluvec/gvpy/python"

// Run a simple stirng.
func Run(cmd string) error {
	gil := GILEnsure()
	defer GILRelease(gil)

	return RunX(cmd)
}

// Lowlevel API: run a simple string.
func RunX(cmd string) error {
	return python.PyRun_SimpleString(cmd)
}
