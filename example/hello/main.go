package main

import (
	"fmt"

	"github.com/bluvec/gvpy"
)

func main() {
	if err := gvpy.Initialize(); err != nil {
		fmt.Println("error to initialize gvpy")
		return
	}
	gvpy.AddSysPath(".")

	fooMod := gvpy.Import("foo")
	if fooMod == nil {
		gvpy.PyErrPrint()
		return
	}

	fooMod.CallFunc("SayHello")
}
