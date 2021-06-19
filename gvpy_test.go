package gvpy

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"testing"
)

func TestMain(m *testing.M) {
	runtime.LockOSThread()
	err := Initialize()
	if err != nil {
		fmt.Println("Error to initialize python")
		os.Exit(1)
	}
	AddSysPath("test")
	tstate := SaveThread()

	rc := m.Run()

	RestoreThread(tstate)
	Finalize()
	runtime.UnlockOSThread()
	os.Exit(rc)
}

func TestImport(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	gil := GILEnsure()
	defer GILRelease(gil)

	fooMod := Import("foo")
	if fooMod == nil {
		PyErrPrint()
		t.Error("error: import foo")
		return
	}
	// This line is optional. Go GC will also handle the clear of fooMod.
	defer fooMod.Clear()

	t.Log(fooMod)
	if fooMod.RefCnt() != 2 {
		t.Errorf("incorrect RefCnt: %v", fooMod.RefCnt())
	}
}

func TestVar(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	gil := GILEnsure()
	defer GILRelease(gil)

	fooMod := Import("foo")
	if fooMod == nil {
		PyErrPrint()
		t.Error("error: import foo")
		return
	}
	t.Log(fooMod)

	ret, err := fooMod.GetVar("num_pups")
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}

	numPups := ret.(int)
	if numPups != 5 {
		t.Error("incorrect num_pups:", numPups)
		return
	}

	err = fooMod.SetVar("num_pups", 6)
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}

	ret, _ = fooMod.GetVar("num_pups")
	newNumPups := ret.(int)
	if newNumPups != 6 {
		t.Error("incorrect newNumPups:", newNumPups)
		return
	}
}

func TestCallFunc(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	t.Parallel()
	gil := GILEnsure()
	defer GILRelease(gil)

	fooFunc := FromImportFunc("foo", "FooFunc")
	if fooFunc == nil {
		PyErrPrint()
		t.Error("error: from foo import FooFunc")
		return
	}
	// This line is optional. Go GC will also handle the clear of fooFunc.
	defer fooFunc.Clear()
	t.Log(fooFunc)

	_, err := fooFunc.Call()
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}

	fooFunc2 := FromImportFunc("foo", "FooFunc2")
	if fooFunc2 != nil {
		t.Error("incorrect: from foo import FooFunc2")
	}
}

func TestCallFuncWithArgs(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	t.Parallel()
	gil := GILEnsure()
	defer GILRelease(gil)

	fooMod := Import("foo")

	name := "Marshal"
	ret, err := fooMod.CallFunc("BarFunc", name)
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}

	msg := ret.(string)
	if msg != "Hi, "+name {
		t.Errorf("incorrect return value: %v\n", msg)
		return
	}
}

func TestClass(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	t.Parallel()
	gil := GILEnsure()
	defer GILRelease(gil)

	fooClass := FromImportClass("foo", "FooClass")
	if fooClass == nil {
		PyErrPrint()
		t.Error("error: from foo import FooClass")
		return
	}
	// This line is optional. Go GC will also handle the clear of fooClass.
	defer fooClass.Clear()
	t.Log(fooClass)

	// static variable
	ret1, err := fooClass.GetVar("prefix")
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}
	prefix, ok := ret1.(string)
	if !ok {
		t.Errorf("returned value type incorrect: %v\n", ret1)
		return
	}
	if prefix != "Paw Patrol" {
		t.Errorf("error to get static var: %v\n", prefix)
		return
	}

	// static method
	ret2, err := fooClass.Call("Prefix")
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}
	if ret2 != "Paw Patrol" {
		t.Errorf("error to call static method: %v\n", ret2)
		return
	}

	// instance
	fooInst := fooClass.New("Rocky")
	if fooInst == nil {
		PyErrPrint()
		t.Error("error: fooInst = FooClass('Rockey')")
		return
	}
	// This line is optional. Go GC will also handle the clear of fooInst.
	defer fooInst.Clear()
	t.Log(fooInst)

	// instance var
	name_, err := fooInst.GetVar("name_")
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}
	if name_ != "Rocky" {
		t.Errorf("error to get instance var: %v\n", name_)
		return
	}

	// instance method: name
	name, err := fooInst.Call("name")
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}
	if name != "Rocky" {
		t.Errorf("error to get instance method: %v", name)
		return
	}
}

func TestListAndDict(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	gil := GILEnsure()
	defer GILRelease(gil)

	fooMod := Import("foo")
	if fooMod == nil {
		PyErrPrint()
		t.Error("error: import foo")
		return
	}

	// prepare a dict
	dict := map[string]int{"Marshal": 1, "Rocky": 2, "Chase": 3}

	// foo.DictKeys
	keyRet, err := fooMod.CallFunc("DictKeys", dict)
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}
	keys, ok := keyRet.([]string)
	if !ok {
		t.Errorf("return value is not a list: %v\n", keyRet)
		return
	}
	if len(keys) != len(dict) {
		t.Error("return list len incorrect")
		return
	}

	for i := 0; i < len(dict); i++ {
		if _, ok := dict[keys[i]]; !ok {
			t.Errorf("return key list incorrect: %v", keys)
			return
		}
	}

	// foo.DictKeysAndValues
	kvRet, err := fooMod.CallFunc("DictKeysAndValues", dict)
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}
	kvRetTup, ok := kvRet.([]interface{})
	if !ok {
		t.Errorf("return type incorrect: %v\n", kvRetTup)
		return
	}

	klist, ok := kvRetTup[0].([]string)
	if !ok {
		t.Errorf("return type incorrect: %v\n", kvRetTup[0])
		return
	}
	vlist, ok := kvRetTup[1].([]int)
	if !ok {
		t.Errorf("return type incorrect: %v\n", kvRetTup[1])
		return
	}

	t.Log(klist)
	t.Log(vlist)
}

func TestNdarray(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	gil := GILEnsure()
	defer GILRelease(gil)

	fooMod := Import("foo")
	if fooMod == nil {
		PyErrPrint()
		t.Error("error: import foo")
		return
	}

	x := make([]int, 10)
	y := make([]int, 10)

	xarr := NewNdarrayFromSlice_int(x, nil)
	if xarr == nil {
		t.Error("error to new ndarray from int slice")
		return
	}

	yarr := NewNdarrayFromSlice_int(y, nil)
	if yarr == nil {
		t.Error("error to new ndarray from int slice")
		return
	}

	if xarr.Dtype() != NPY_INT {
		t.Errorf("dtype incorrect: %v\n", xarr.Dtype())
		return
	}

	zarrRet, err := fooMod.CallFunc("NdarrayAdd", xarr, yarr)
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}
	zarr, ok := zarrRet.(Ndarray)
	if !ok {
		t.Error("error to cast ndarray")
		return
	}
	if zarr.Dtype() != NPY_INT {
		t.Errorf("dtype of returned ndarray is incorrect: %v\n", zarr.Dtype())
		return
	}

	z := zarr.AsSlice_int()
	if len(z) != len(x) {
		t.Errorf("len of returned ndarray incorrect: %v\n", len(z))
		return
	}

	for i := 0; i < len(x); i++ {
		if z[i] != x[i]+y[i] {
			t.Error("incorrect return values")
			return
		}
	}
}

func TestNdarray2(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	gil := GILEnsure()
	defer GILRelease(gil)

	fooMod := Import("foo")
	if fooMod == nil {
		PyErrPrint()
		t.Error("error: import foo")
		return
	}

	x := make([]complex64, 10)
	y := make([]complex64, 10)

	for i := 0; i < len(x); i++ {
		x[i] = complex(rand.Float32(), rand.Float32())
		y[i] = complex(rand.Float32(), rand.Float32())
	}

	xarr := NewNdarrayFromSlice_complex64(x, []int{2, 5})
	if xarr == nil {
		t.Error("error to new ndarray from int slice")
		return
	}

	yarr := NewNdarrayFromSlice_complex64(y, []int{2, 5})
	if yarr == nil {
		t.Error("error to new ndarray from int slice")
		return
	}

	if xarr.Dtype() != NPY_COMPLEX64 {
		t.Errorf("dtype incorrect: %v\n", xarr.Dtype())
		return
	}

	zarrRet, err := fooMod.CallFunc("NdarrayAdd", xarr, yarr)
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}
	zarr, ok := zarrRet.(Ndarray)
	if !ok {
		t.Error("error to cast ndarray")
		return
	}
	if zarr.Dtype() != NPY_COMPLEX64 {
		t.Errorf("dtype of returned ndarray is incorrect: %v\n", zarr.Dtype())
		return
	}

	shape := zarr.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 5 {
		t.Errorf("incorrect return shape: %v\n", shape)
		return
	}

	z := zarr.AsSlice_complex64()
	if len(z) != len(x) {
		t.Errorf("len of returned ndarray incorrect: %v\n", len(z))
		return
	}

	for i := 0; i < len(x); i++ {
		if z[i] != x[i]+y[i] {
			t.Error("incorrect return values")
			return
		}
	}
}

func TestRunSimpleString(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	gil := GILEnsure()
	defer GILRelease(gil)

	cmd := "import foo; foo.RunSimpleString()"
	if err := Run(cmd); err != nil {
		PyErrPrint()
		t.Error(err)
	}
}
