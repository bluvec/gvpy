package gvpy

import (
	"fmt"
	"testing"
)

func TestInit(t *testing.T) {
	fmt.Println("=================================")
	fmt.Println("TestInit")

	Py_Initialize()
	if !Py_IsInitialized() {
		t.Error("error to initialize python interpreter")
		PyErr_Print()
		return
	}
	Py_Finalize()

	fmt.Println("=================================")
}

func TestVersion(t *testing.T) {
	fmt.Println("=================================")
	fmt.Println("TestVersion")

	Py_Initialize()
	if !Py_IsInitialized() {
		t.Error("error to initialize python interpreter")
		PyErr_Print()
		return
	}
	defer Py_Finalize()

	// Get and set paths
	paths := Py_GetPath()
	fmt.Println(len(paths))

	// Get version
	pyVer := Py_GetVersion()
	fmt.Println(pyVer)

	fmt.Println("=================================")
}

func TestImport(t *testing.T) {
	fmt.Println("=================================")
	fmt.Println("TestImport")

	Py_Initialize()
	defer Py_Finalize()

	// import foo
	fooMod := PyImport_ImportModule("foo")
	if fooMod == nil {
		t.Error("failed to import module 'foo'")
		PyErr_Print()
		return
	}
	defer fooMod.Py_Clear()
	fmt.Println(fooMod)

	fmt.Println("=================================")
}

func TestFooFunc(t *testing.T) {
	fmt.Println("=================================")
	fmt.Println("TestFooFunc")

	Py_Initialize()
	defer Py_Finalize()

	// import foo
	fmt.Println("---------------------------------")
	fooMod := PyImport_ImportModule("foo")
	if fooMod == nil {
		t.Error("failed to import module 'foo'")
		return
	}
	defer fooMod.Py_Clear()
	fmt.Println(fooMod)

	// FooFunc
	fmt.Println("---------------------------------")
	fooFunc := fooMod.GetAttrString("FooFunc")
	if fooFunc == nil {
		t.Error("error to get func 'FooFunc'")
		return
	}
	defer fooFunc.Py_Clear()
	fmt.Println(fooFunc)
	fmt.Println("fooFunc is function: ", PyFunction_Check(fooFunc))
	fmt.Println("fooFunc is callable: ", fooFunc.Callable())

	fmt.Println("---------------------------------")
	fooArgs := PyTuple_New(0)
	if fooArgs == nil {
		t.Error("error to new fooArgs")
		return
	}
	defer fooArgs.Py_Clear()

	fooRet := fooFunc.Call(fooArgs, nil)
	if fooRet == nil {
		t.Error("error to call fooFunc")
		return
	}
	defer fooRet.Py_Clear()

	fmt.Println("=================================")
}

func TestBarFunc(t *testing.T) {
	fmt.Println("=================================")
	fmt.Println("TestBarFunc")

	Py_Initialize()
	defer Py_Finalize()

	// import foo
	fmt.Println("---------------------------------")
	fooMod := PyImport_ImportModule("foo")
	if fooMod == nil {
		t.Error("failed to import module 'foo'")
		return
	}
	defer fooMod.Py_Clear()
	fmt.Println(fooMod)

	// BarFunc
	fmt.Println("---------------------------------")
	barFunc := fooMod.GetAttrString("BarFunc")
	if barFunc == nil {
		t.Error("error to get func 'BarFunc'")
		return
	}
	defer barFunc.Py_Clear()
	fmt.Println(barFunc)
	fmt.Println("barFunc is function: ", PyFunction_Check(barFunc))
	fmt.Println("barFunc is callable: ", barFunc.Callable())

	fmt.Println("---------------------------------")
	barArgs := PyTuple_New(1)
	if barArgs == nil {
		t.Error("error to new barArgs")
		return
	}
	defer barArgs.Py_Clear()

	err := PyTuple_SetItem(barArgs, 0, PyUnicode_FromString("Marshal"))
	if err != nil {
		t.Error("error to set barArgs")
		return
	}

	barRet := barFunc.Call(barArgs, nil)
	if barRet == nil {
		t.Error("error to call barFunc")
		return
	}
	defer barRet.Py_Clear()

	fmt.Println("barRet is unicode: ", PyUnicode_Check(barRet))
	barStr := PyUnicode_AsString(barRet)
	fmt.Println(barStr)

	fmt.Println("=================================")
}

func TestFooClass(t *testing.T) {
	fmt.Println("=================================")
	fmt.Println("TestFooClass")

	Py_Initialize()
	defer Py_Finalize()

	// import foo
	fmt.Println("---------------------------------")
	fooMod := PyImport_ImportModule("foo")
	if fooMod == nil {
		t.Error("failed to import module 'foo'")
		PyErr_Print()
		return
	}
	defer fooMod.Py_Clear()
	fmt.Println(fooMod)

	// class Foo
	fmt.Println("---------------------------------")
	fooClass := fooMod.GetAttrString("FooClass")
	if fooClass == nil {
		t.Error("error to get class 'FooClass'")
		return
	}
	defer fooClass.Py_Clear()
	fmt.Println(fooClass)
	fmt.Println("fooClass is callable: ", fooClass.Callable())

	// instance of Foo
	fmt.Println("---------------------------------")
	arg1 := PyTuple_New(0)
	defer arg1.Py_Clear()

	fooInst := fooClass.Call(arg1, nil)
	if fooInst == nil {
		t.Error("error to instantiate FooClass")
		PyErr_Print()
		return
	}
	defer fooInst.Py_Clear()
	fmt.Println(fooInst)
	fmt.Println("fooInst is callable: ", fooInst.Callable())

	// method: Foo.print
	fmt.Println("---------------------------------")
	printMeth := fooInst.GetAttrString("print")
	if printMeth == nil {
		t.Error("error to get method 'FooClass.print'")
		PyErr_Print()
		return
	}
	defer printMeth.Py_Clear()
	fmt.Println(printMeth)
	fmt.Println("printMeth is callable: ", printMeth.Callable())

	printRet := printMeth.Call(arg1, nil)
	if printRet == nil {
		t.Error("error to call printMeth")
		return
	}

	// method: Foo.name
	fmt.Println("---------------------------------")
	nameMeth := fooInst.GetAttrString("name")
	if nameMeth == nil {
		t.Error("error to get method 'FooClass.nameMeth'")
		PyErr_Print()
		return
	}
	defer nameMeth.Py_Clear()
	fmt.Println(nameMeth)
	fmt.Println("nameMeth is callable: ", nameMeth.Callable())

	nameRet := nameMeth.Call(arg1, nil)
	if nameRet == nil {
		t.Error("error to call nameMeth")
		return
	}
	defer nameRet.Py_Clear()

	name := PyUnicode_AsString(nameRet)
	fmt.Println("Gvpy: ", name)
	if name != "Rubble" {
		t.Error("return incorrect value from nameMeth")
		return
	}

	fmt.Println("=================================")
}

func TestBarClass(t *testing.T) {
	fmt.Println("=================================")
	fmt.Println("TestFooClass")

	Py_Initialize()
	defer Py_Finalize()

	// import foo
	fmt.Println("---------------------------------")
	fooMod := PyImport_ImportModule("foo")
	if fooMod == nil {
		t.Error("failed to import module 'foo'")
		PyErr_Print()
		return
	}
	defer fooMod.Py_Clear()
	fmt.Println(fooMod)

	// class Bar
	fmt.Println("---------------------------------")
	barClass := fooMod.GetAttrString("BarClass")
	if barClass == nil {
		t.Error("error to get class 'BarClass'")
		PyErr_Print()
		return
	}
	defer barClass.Py_Clear()
	fmt.Println(barClass)
	fmt.Println("barClass is callable: ", barClass.Callable())

	// instance of Bar
	fmt.Println("---------------------------------")
	arg1 := PyTuple_New(1)
	if arg1 == nil {
		t.Error("error to create arg1")
		PyErr_Print()
		return
	}
	defer arg1.Py_Clear()

	err := PyTuple_SetItem(arg1, 0, PyUnicode_FromString("Chase"))
	if err != nil {
		t.Error("error to set tuple")
		PyErr_Print()
		return
	}

	barInst := barClass.Call(arg1, nil)
	if barInst == nil {
		t.Error("error to instantiate BarClass")
		PyErr_Print()
		return
	}
	defer barInst.Py_Clear()

	// BarClass.print
	fmt.Println("---------------------------------")
	printMeth := barInst.GetAttrString("print")
	if printMeth == nil {
		t.Error("error to get BarClass.print")
		PyErr_Print()
		return
	}
	defer printMeth.Py_Clear()
	fmt.Println(printMeth)

	arg2 := PyTuple_New(1)
	defer arg2.Py_Clear()
	PyTuple_SetItem(arg2, 0, PyUnicode_FromString("Hi,"))

	printRet := printMeth.Call(arg2, nil)
	if printRet == nil {
		t.Error("error to call BarClass.print")
		PyErr_Print()
		return
	}

	// BarClass.name
	fmt.Println("---------------------------------")

	nameMeth := barInst.GetAttrString("name")
	if nameMeth == nil {
		t.Error("error to get method BarClass.name")
		return
	}
	defer nameMeth.Py_Clear()
	fmt.Println(nameMeth)

	arg3 := PyTuple_New(0)
	defer arg3.Py_Clear()

	nameRet := nameMeth.Call(arg3, nil)
	if nameRet == nil {
		t.Error("eror to call BarClass.name")
		PyErr_Print()
		return
	}
	defer nameRet.Py_Clear()

	fmt.Println("=================================")
}

func TestBytes(t *testing.T) {
	fmt.Println("=================================")
	fmt.Println("TestBytes")

	Py_Initialize()
	defer Py_Finalize()

	fooMod := PyImport_ImportModule("foo")
	if fooMod == nil {
		t.Errorf("error to import module 'foo'")
		return
	}
	defer fooMod.Py_Clear()

	showBytesFunc := fooMod.GetAttrString("ShowBytes")
	if showBytesFunc == nil {
		t.Errorf("error to get func 'BytesAdd'")
		return
	}
	defer showBytesFunc.Py_Clear()

	args := PyTuple_New(1)
	defer args.Py_Clear()

	x := []byte{49, 50, 51, 52, 53, 54, 0}
	xobj := PyBytes_FromBytes(x)
	PyTuple_SetItem(args, 0, xobj)

	zObj := showBytesFunc.Call(args, nil)
	if zObj == nil {
		t.Error("error to call BytesAdd function")
		return
	}
	defer zObj.Py_Clear()

	if !PyBytes_Check(zObj) {
		t.Errorf("return value is not a byte array")
		return
	}
	z := PyBytes_AsBytes(zObj)
	if len(z) != len(x) {
		t.Errorf("return value length error")
		return
	}
	for i := 0; i < len(z); i++ {
		if z[i] != x[i] {
			t.Errorf("return value incorrect error")
			return
		}
	}
	fmt.Println(z)

	fmt.Println("=================================")
}

func TestGetPath(t *testing.T) {
	fmt.Println("=================================")
	fmt.Println("TestGetPath")

	Py_Initialize()
	defer Py_Finalize()

	sysPath := Py_GetPath()
	if sysPath == "" {
		t.Error("error to get path")
		return
	}
	fmt.Println(sysPath)

	fmt.Println("=================================")
}

func testSetPath(t *testing.T) {
	fmt.Println("=================================")
	fmt.Println("TestSetPath")

	Py_Initialize()
	defer Py_Finalize()

	p1 := Py_GetPath()
	if p1 == "" {
		t.Error("error to get path")
		return
	}
	fmt.Println("p1: ", p1)

	p2 := "/opt:/bin:/usr"
	Py_SetPath(p2)

	p3 := Py_GetPath()
	if p3 == "" {
		t.Error("error to get path")
		return
	}
	fmt.Println("p3: ", p3)

	if p3 != p2 {
		t.Error("paths unequal error")
		return
	}

	fmt.Println("=================================")
}
