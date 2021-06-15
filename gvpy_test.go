package gvpy

import (
	"fmt"
	"gvpy/python"
	"testing"
)

func TestGvpy(t *testing.T) {
	fmt.Println("=================================")
	fmt.Println("Initialize")
	initialize(t)

	fmt.Println("=================================")
	fmt.Println("testImport")
	testImport(t)

	fmt.Println("=================================")
	fmt.Println("testFooFunc")
	testFooFunc(t)

	fmt.Println("=================================")
	fmt.Println("testBarFunc")
	testBarFunc(t)

	fmt.Println("=================================")
	fmt.Println("testBarNumPups")
	testBarNumPups(t)

	fmt.Println("=================================")
	fmt.Println("testBarClass")
	testBarClass(t)

	fmt.Println("=================================")
	fmt.Println("testBytes")
	testBytes(t)

	fmt.Println("=================================")
	fmt.Println("testDict")
	testDict(t)

	fmt.Println("=================================")
	fmt.Println("testNdarray")
	testNdarray(t)

	fmt.Println("=================================")
	fmt.Println("Finalize")
	finalize(t)
}

func initialize(t *testing.T) {
	Initialize()
	AddSysPathAtFront("./test")
}

func finalize(t *testing.T) {
	Finalize()
}

func testImport(t *testing.T) {
	barMod, err := Import("bar")
	if err != nil {
		PyErrPrint()
		t.Errorf("%+v", err)
		return
	}
	defer barMod.Clear()
	fmt.Println(barMod)
}

func testFooFunc(t *testing.T) {
	fooFunc, err := FromImportFunc("bar", "FooFunc")
	if err != nil {
		t.Error(err)
		return
	}
	fmt.Println(fooFunc)

	_, err = fooFunc.Call()
	if err != nil {
		t.Error(err)
		return
	}
}

func testBarFunc(t *testing.T) {
	barMod, err := Import("bar")
	if err != nil {
		t.Error(err)
		return
	}
	defer barMod.Clear()

	barFunc, err := barMod.GetFunc("BarFunc")
	if err != nil {
		t.Error(err)
		return
	}
	defer barFunc.Clear()

	barRet, err := barFunc.Call("Marshal")
	if err != nil {
		t.Errorf("error to call BarFunc: %v", err)
		return
	}
	if barRet != "Hi, Marshal" {
		t.Errorf("return value incorrect: %v", barRet)
		return
	}
	fmt.Println(barRet)
}

func testBarNumPups(t *testing.T) {
	barMod, err := Import("bar")
	if err != nil {
		t.Error(err)
		return
	}
	defer barMod.Clear()

	numPups, err := barMod.GetVar("num_pups")
	if err != nil {
		t.Error(err)
		return
	}
	if numPups != 5 {
		t.Errorf("num_pups: %v", numPups)
		return
	}
	fmt.Println("numPups: ", numPups)

	err = barMod.SetVar("num_pups", 6)
	if err != nil {
		t.Error(err)
		return
	}

	newNumPups, err := barMod.GetVar("num_pups")
	if err != nil {
		t.Error(err)
		return
	}

	if newNumPups != 6 {
		t.Errorf("new_num_pups: %v", numPups)
	}
	fmt.Println("newNumPups: ", newNumPups)
}

func testBarClass(t *testing.T) {
	barMod, err := Import("bar")
	if err != nil {
		t.Error(err)
		return
	}
	defer barMod.Clear()

	barClass, err := barMod.GetClass("BarClass")
	if err != nil {
		t.Error(err)
		return
	}
	defer barClass.Clear()

	// static variable
	staticVar, err := barClass.GetVar("prefix")
	if err != nil {
		t.Error(err)
		return
	}
	if staticVar != "Hi," {
		t.Errorf("error to get static var: '%v'", staticVar)
		return
	}

	// static method
	staticMethodRet, err := barClass.Call("Prefix")
	if err != nil {
		t.Error(err)
		return
	}
	if staticMethodRet != "Hi," {
		t.Errorf("error to get static method: '%v'", staticMethodRet)
	}
	fmt.Println(staticMethodRet)

	// instance
	barInst, err := barClass.New("Rocky")
	if err != nil {
		t.Error(err)
		return
	}
	defer barInst.Clear()
	fmt.Println(barInst)

	// instance var
	instVar, err := barInst.GetVar("name_")
	if err != nil {
		t.Error(err)
		return
	}
	if instVar != "Rocky" {
		t.Errorf("error to get instance var: '%v'", instVar)
		return
	}
	fmt.Println(instVar)

	// instance method: name
	nameRet, err := barInst.Call("name")
	if err != nil {
		t.Error(err)
		return
	}
	if nameRet != "Rocky" {
		t.Errorf("error to get instance method: '%v'", nameRet)
		return
	}

	// instance method: print
	_, err = barInst.Call("print", "!")
	if err != nil {
		t.Error(err)
		return
	}
}

func testBytes(t *testing.T) {
	// Import module
	fmt.Println("---------------------------------")
	barMod, err := Import("bar")
	if err != nil {
		t.Errorf("%+v", err)
		return
	}
	defer barMod.Clear()
	fmt.Println(barMod)

	// python: bar.ShowBytes
	fmt.Println("---------------------------------")
	x := []byte{49, 50, 51, 52, 53, 54, 55}
	ret, err := barMod.CallFunc("ShowBytes", x)
	if err != nil {
		t.Errorf("error to call func: ShowBytes, err: %v", err)
		return
	}
	retSlice := ret.([]byte)
	y := string(retSlice)
	if y != "1234567" {
		t.Errorf("return value incorrect: %v", y)
		return
	}
	fmt.Println(y)
}

func testDict(t *testing.T) {
	barMod, err := Import("bar")
	if err != nil {
		t.Error(err)
		return
	}
	defer barMod.Clear()

	// python: bar.DictKeys
	fmt.Println("---------------------------------")
	dict := map[string]int{"a": 1, "b": 2, "c": 3}
	keysRet, err := barMod.CallFunc("DictKeys", dict)
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}
	fmt.Println(keysRet)
	keys := keysRet.([]string)
	for i := 0; i < 3; i++ {
		if _, ok := dict[keys[i]]; !ok {
			t.Errorf("error to call bar.DictKeys: '%v'", keys)
			return
		}
	}

	// python: bar.DictKeysAndValues
	fmt.Println("---------------------------------")
	kvRet, err := barMod.CallFunc("DictKeysAndValues", dict)
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}
	kvRetTup := kvRet.([]interface{})
	klist := kvRetTup[0].([]string)
	vlist := kvRetTup[1].([]int)
	fmt.Println(klist)
	fmt.Println(vlist)
}

func testNdarray(t *testing.T) {
	barMod, err := Import("bar")
	if err != nil {
		t.Error(err)
		return
	}
	defer barMod.Clear()

	// python: GetNdarray
	fmt.Println("---------------------------------")

	getNdarrayFunc, err := barMod.GetFunc("GetNdarray")
	if err != nil {
		PyErrPrint()
		t.Errorf("error to get func: GetNdarray")
		return
	}
	defer getNdarrayFunc.Clear()

	// get a new ndarray
	ndarrayObj1, err := call(getNdarrayFunc.pyobj)
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}
	defer ndarrayObj1.Py_Clear()
	fmt.Println(ndarrayObj1)

	// check ndarray type
	if !python.PyArray_Check(ndarrayObj1) {
		PyErrPrint()
		t.Error("ndarrayObj1 is not an PyArrayObject")
		return
	}

	ndim := python.PyArray_NDIM(ndarrayObj1)
	if ndim != 2 {
		PyErrPrint()
		t.Errorf("error to get ndim. ndim: %v", ndim)
		return
	}

	dtype := python.PyArray_TYPE(ndarrayObj1)
	fmt.Println(dtype)

	// python: NdarrayAdd
	fmt.Println("---------------------------------")

	data1 := []byte{1, 2, 3, 4, 5, 6}
	ndarr1 := NewNdarrayFromSlice_byte(data1, []int{2, 3})
	fmt.Println(ndarr1)
	fmt.Println(ndarr1.Shape())

	data2 := []int{21, 22, 23, 24, 25, 26}
	ndarr2 := NewNdarrayFromSlice_int(data2, nil)
	item2 := ndarr2.AsSlice_int()
	for i := range data2 {
		if data2[i] != item2[i] {
			t.Errorf("data2 != item2 error. data2: %v, item2: %v", data2, item2)
			return
		}
	}

	data3 := []float64{10, 20, 30, 40}
	ndarr3 := NewNdarrayFromSlice_float64(data3, nil)
	item3 := ndarr3.AsSlice_float64()
	for i := range data3 {
		if data3[i] != item3[i] {
			t.Errorf("data3 != item3 error. data3: %v, item3: %v", data3, item3)
			return
		}
	}

	ret3, err := barMod.CallFunc("NdarrayAdd", ndarr3, 100)
	if err != nil {
		PyErrPrint()
		t.Error(err)
		return
	}
	retArr3 := ret3.(Ndarray)
	fmt.Println(retArr3.Shape())
	fmt.Println(retArr3.Dtype())
}
