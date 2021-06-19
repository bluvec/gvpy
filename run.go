package gvpy

import "github.com/bluvec/gvpy/python"

func Run(cmd string) error {
	return python.PyRun_SimpleString(cmd)
}
