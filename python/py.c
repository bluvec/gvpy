#include "py.h"
#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <stdio.h>

// patchlevel
char *__Py_VERSION() {
  return PY_VERSION;
}

int __Py_VERSION_HEX() {
  return PY_VERSION_HEX;
}

int __Py_MAJOR_VERSION() {
  return PY_MAJOR_VERSION;
}

int __Py_MINOR_VERSION() {
  return PY_MINOR_VERSION;
}

int __Py_MICRO_VERSION() {
  return PY_MICRO_VERSION;
}

// error
PyObject *__PyErr_Occurred() {
  return PyErr_Occurred();
}

void __PyErr_PrintEx(int set_sys_last_vars) {
  PyErr_PrintEx(set_sys_last_vars);
}

void __PyErr_Print() {
  PyErr_Print();
}

void __PyErr_Clear() {
  PyErr_Clear();
}

// lifecycle
void __Py_Initialize() {
  Py_Initialize();
}

void __Py_InitializeEx(int initsigs) {
  Py_InitializeEx(initsigs);
}

int __Py_IsInitialized() {
  return Py_IsInitialized();
}

int __Py_FinalizeEx() {
  return Py_FinalizeEx();
}

char *__Py_GetPath() {
  wchar_t *wpath = Py_GetPath();

  size_t err_pos;
  char *path = Py_EncodeLocale(wpath, &err_pos);
  (void)err_pos;

  return path;
}

void __Py_SetPath(const char *path) {
  size_t size;
  wchar_t *wpath = Py_DecodeLocale(path, &size);
  (void)size;

  Py_SetPath(wpath);

  PyMem_RawFree(wpath);
}

const char *__Py_GetVersion() {
  return Py_GetVersion();
}

// pymem
void *__PyMem_Malloc(size_t n) {
  return PyMem_Malloc(n);
}

void *__PyMem_Realloc(void *p, size_t n) {
  return PyMem_Realloc(p, n);
}

void __PyMem_Free(void *p) {
  PyMem_Free(p);
}

// pystate
int __PyGILState_Check() {
  return PyGILState_Check();
}

__PyGILState_STATE __PyGILState_Ensure() {
  return (__PyGILState_STATE)PyGILState_Ensure();
}

void __PyGILState_Release(__PyGILState_STATE s) {
  PyGILState_Release((PyGILState_STATE)s);
}

// ceval
int __PyEval_ThreadsInitialized() {
#if PY_VERSION_HEX < 0x03090000
  return PyEval_ThreadsInitialized();
#else
  return 0;
#endif
}

void __PyEval_InitThreads() {
#if PY_VERSION_HEX < 0x03090000
  PyEval_InitThreads();
#endif
}

PyThreadState *__PyEval_SaveThread() {
  return PyEval_SaveThread();
}

void __PyEval_RestoreThread(PyThreadState *tstate) {
  PyEval_RestoreThread(tstate);
}

// pythonrun
int __PyRun_SimpleString(const char *command) {
  return PyRun_SimpleString(command);
}

// object
PyObject *__Py_None() {
  return Py_None;
}

void __Py_CLEAR(PyObject *o) {
  Py_CLEAR(o);
}

void __Py_XINCREF(PyObject *o) {
  Py_XINCREF(o);
}

void __Py_XDECREF(PyObject *o) {
  Py_XDECREF(o);
}

Py_ssize_t __Py_REFCNT(PyObject *o) {
  return Py_REFCNT(o);
}

PyTypeObject *__Py_TYPE(PyObject *o) {
  return Py_TYPE(o);
}

int __PyObject_IsInstance(PyObject *p, PyObject *c) {
  return PyObject_IsInstance(p, c);
}

PyObject *__PyObject_Str(PyObject *o) {
  return PyObject_Str(o);
}

int __PyObject_HasAttrString(PyObject *o, const char *attr_name) {
  return PyObject_HasAttrString(o, attr_name);
}

PyObject *__PyObject_GetAttrString(PyObject *o, const char *attr_name) {
  return PyObject_GetAttrString(o, attr_name);
}

int __PyObject_SetAttrString(PyObject *o, const char *attr_name, PyObject *v) {
  return PyObject_SetAttrString(o, attr_name, v);
}

int __PyCallable_Check(PyObject *o) {
  return PyCallable_Check(o);
}

PyObject *__PyObject_Call(PyObject *p, PyObject *args, PyObject *kwargs) {
  return PyObject_Call(p, args, kwargs);
}

PyObject *__PyObject_CallObject(PyObject *p, PyObject *args) {
  return PyObject_CallObject(p, args);
}

// import
PyObject *__PyImport_ImportModule(const char *name) {
  return PyImport_ImportModule(name);
}

PyObject *__PyImport_ImportModuleLevel(const char *name, PyObject *globals,
                                       PyObject *locals, PyObject *fromlist,
                                       int level) {
  return PyImport_ImportModuleLevel(name, globals, locals, fromlist, level);
}

// sysmodule
PyObject *__PySys_GetObject(const char *name) {
  return PySys_GetObject(name);
}

int __PySys_SetObject(const char *name, PyObject *v) {
  return PySys_SetObject(name, v);
}

// boolobject
PyObject *__Py_True() {
  return Py_True;
}

PyObject *__Py_False() {
  return Py_False;
}

int __PyBool_Check(PyObject *o) {
  return PyBool_Check(o);
}

PyObject *__PyBool_FromLong(long v) {
  return PyBool_FromLong(v);
}

// longobject
int __PyLong_Check(PyObject *o) {
  return PyLong_Check(o);
}

long __PyLong_AsLong(PyObject *obj) {
  return PyLong_AsLong(obj);
}

PyObject *__PyLong_FromLong(long v) {
  return PyLong_FromLong(v);
}

// floatobject
int __PyFloat_Check(PyObject *o) {
  return PyFloat_Check(o);
}

double __PyFloat_AsDouble(PyObject *pyfloat) {
  return PyFloat_AsDouble(pyfloat);
}

PyObject *__PyFloat_FromDouble(double v) {
  return PyFloat_FromDouble(v);
}

// bytesobject
int __PyBytes_Check(PyObject *o) {
  return PyBytes_Check(o);
}

PyObject *__PyBytes_FromString(const char *v) {
  return PyBytes_FromString(v);
}

PyObject *__PyBytes_FromStringAndSize(const char *v, Py_ssize_t len) {
  return PyBytes_FromStringAndSize(v, len);
}

Py_ssize_t __PyBytes_Size(PyObject *o) {
  return PyBytes_Size(o);
}

char *__PyBytes_AsString(PyObject *o) {
  return PyBytes_AsString(o);
}

// listobject
int __PyList_Check(PyObject *o) {
  return PyList_Check(o);
}

PyObject *__PyList_New(Py_ssize_t len) {
  return PyList_New(len);
}

Py_ssize_t __PyList_Size(PyObject *list) {
  return PyList_Size(list);
}

PyObject *__PyList_GetItem(PyObject *list, Py_ssize_t index) {
  return PyList_GetItem(list, index);
}

int __PyList_SetItem(PyObject *list, Py_ssize_t index, PyObject *item) {
  return PyList_SetItem(list, index, item);
}

int __PyList_Insert(PyObject *list, Py_ssize_t index, PyObject *item) {
  return PyList_Insert(list, index, item);
}

int __PyList_Append(PyObject *list, PyObject *item) {
  return PyList_Append(list, item);
}

PyObject *__PyList_AsTuple(PyObject *list) {
  return PyList_AsTuple(list);
}

// dictobject
int __PyDict_Check(PyObject *o) {
  return PyDict_Check(o);
}

PyObject *__PyDict_New() {
  return PyDict_New();
}

PyObject *__PyDict_GetItem(PyObject *p, PyObject *key) {
  return PyDict_GetItem(p, key);
}

PyObject *__PyDict_GetItemString(PyObject *p, const char *key) {
  return PyDict_GetItemString(p, key);
}

int __PyDict_SetItem(PyObject *p, PyObject *key, PyObject *val) {
  return PyDict_SetItem(p, key, val);
}

int __PyDict_SetItemString(PyObject *p, const char *key, PyObject *val) {
  return PyDict_SetItemString(p, key, val);
}

int __PyDict_DelItem(PyObject *p, PyObject *key) {
  return PyDict_DelItem(p, key);
}

int __PyDict_DelItemString(PyObject *p, const char *key) {
  return PyDict_DelItemString(p, key);
}

int __PyDict_Contains(PyObject *p, PyObject *key) {
  return PyDict_Contains(p, key);
}

void __PyDict_Clear(PyObject *p) {
  PyDict_Clear(p);
}

PyObject *__PyDict_Items(PyObject *p) {
  return PyDict_Items(p);
}

PyObject *__PyDict_Keys(PyObject *p) {
  return PyDict_Keys(p);
}

PyObject *__PyDict_Values(PyObject *p) {
  return PyDict_Values(p);
}

Py_ssize_t __PyDict_Size(PyObject *p) {
  return PyDict_Size(p);
}

int __PyDict_Next(PyObject *p, Py_ssize_t *ppos, PyObject **pkey,
                  PyObject **pvalue) {
  return PyDict_Next(p, ppos, pkey, pvalue);
}

int __PyDict_Merge(PyObject *a, PyObject *b, int override) {
  return PyDict_Merge(a, b, override);
}

int __PyDict_Update(PyObject *a, PyObject *b) {
  return PyDict_Update(a, b);
}

// tupleobject
int __PyTuple_Check(PyObject *o) {
  return PyTuple_Check(o);
}

PyObject *__PyTuple_New(Py_ssize_t len) {
  return PyTuple_New(len);
}

Py_ssize_t __PyTuple_Size(PyObject *p) {
  return PyTuple_Size(p);
}

PyObject *__PyTuple_GetItem(PyObject *p, Py_ssize_t pos) {
  return PyTuple_GetItem(p, pos);
}

int __PyTuple_SetItem(PyObject *p, Py_ssize_t pos, PyObject *o) {
  return PyTuple_SetItem(p, pos, o);
}

// funcobject
int __PyFunction_Check(PyObject *o) {
  return PyFunction_Check(o);
}

// unicodeobject
int __PyUnicode_Check(PyObject *o) {
  return PyUnicode_Check(o);
}

PyObject *__PyUnicode_FromString(const char *u) {
  return PyUnicode_FromString(u);
}

const char *__PyUnicode_AsUTF8(PyObject *unicode) {
  return PyUnicode_AsUTF8(unicode);
}

// moduleobject
int __PyModule_Check(PyObject *o) {
  return PyModule_Check(o);
}

// classobject
int __PyClass_Check(PyObject *o) {
  return PyObject_IsInstance(o, (PyObject *)&PyType_Type);
}

// arrayobject
int __PyArray_import_array() {
  if (PyArray_API == NULL) {
    return _import_array();
  }

  return 0;
}

int __PyArray_Check(PyObject *o) {
  return PyArray_Check(o);
}

int __PyArray_NDIM(PyObject *arr) {
  return PyArray_NDIM((PyArrayObject *)arr);
}

int __PyArray_FLAGS(PyObject *arr) {
  return PyArray_FLAGS((PyArrayObject *)arr);
}

int __PyArray_TYPE(PyObject *arr) {
  return PyArray_TYPE((PyArrayObject *)arr);
}

int __PyArray_SETITEM(PyObject *arr, void *itemptr, PyObject *obj) {
  return PyArray_SETITEM((PyArrayObject *)arr, itemptr, obj);
}

void *__PyArray_DATA(PyObject *arr) {
  return PyArray_DATA((PyArrayObject *)arr);
}

char *__PyArray_BYTES(PyObject *arr) {
  return PyArray_BYTES((PyArrayObject *)arr);
}

npy_intp *__PyArray_DIMS(PyObject *arr) {
  return PyArray_DIMS((PyArrayObject *)arr);
}

npy_intp *__PyArray_SHAPE(PyObject *arr) {
  return PyArray_SHAPE((PyArrayObject *)arr);
}

npy_intp *__PyArray_STRIDES(PyObject *arr) {
  return PyArray_STRIDES((PyArrayObject *)arr);
}

npy_intp __PyArray_DIM(PyObject *arr, int n) {
  return PyArray_DIM((PyArrayObject *)arr, n);
}

npy_intp __PyArray_STRIDE(PyObject *arr, int n) {
  return PyArray_STRIDE((PyArrayObject *)arr, n);
}

npy_intp __PyArray_ITEMSIZE(PyObject *arr) {
  return PyArray_ITEMSIZE((PyArrayObject *)arr);
}

npy_intp __PyArray_SIZE(PyObject *arr) {
  return PyArray_SIZE((PyArrayObject *)arr);
}

npy_intp __PyArray_Size(PyObject *obj) {
  return PyArray_Size(obj);
}

npy_intp __PyArray_NBYTES(PyObject *arr) {
  return PyArray_NBYTES((PyArrayObject *)arr);
}

PyObject *__PyArray_BASE(PyObject *arr) {
  return PyArray_BASE((PyArrayObject *)arr);
}

PyArray_Descr *__PyArray_DESCR(PyObject *arr) {
  return PyArray_DESCR((PyArrayObject *)arr);
}

PyArray_Descr *__PyArray_DTYPE(PyObject *arr) {
  return PyArray_DTYPE((PyArrayObject *)arr);
}

PyObject *__PyArray_GETITEM(PyObject *arr, void *itemptr) {
  return PyArray_GETITEM((PyArrayObject *)arr, itemptr);
}

void *__PyArray_GetPtr(PyObject *aobj, npy_intp *ind) {
  return PyArray_GetPtr((PyArrayObject *)aobj, ind);
}

void *__PyArray_GETPTR1(PyObject *obj, npy_intp i) {
  return PyArray_GETPTR1((PyArrayObject *)obj, i);
}

void *__PyArray_GETPTR2(PyObject *obj, npy_intp i, npy_intp j) {
  return PyArray_GETPTR2((PyArrayObject *)obj, i, j);
}

void *__PyArray_GETPTR3(PyObject *obj, npy_intp i, npy_intp j, npy_intp k) {
  return PyArray_GETPTR3((PyArrayObject *)obj, i, j, k);
}

void *__PyArray_GETPTR4(PyObject *obj, npy_intp i, npy_intp j, npy_intp k,
                        npy_intp l) {
  return PyArray_GETPTR4((PyArrayObject *)obj, i, j, k, l);
}

PyObject *__PyArray_SimpleNew(int nd, npy_intp *dims, int typenum) {
  return PyArray_SimpleNew(nd, dims, typenum);
}

PyObject *__PyArray_SimpleNewFromData(int nd, npy_intp const *dims, int typenum,
                                      void *data) {
  return PyArray_SimpleNewFromData(nd, (npy_intp *)dims, typenum, data);
}

PyObject *__PyArray_Zeros(int nd, npy_intp const *dims, PyArray_Descr *dtype,
                          int fortran) {
  return PyArray_Zeros(nd, (npy_intp *)dims, dtype, fortran);
}

PyObject *__PyArray_ZEROS(int nd, npy_intp const *dims, int type_num,
                          int fortran) {
  return PyArray_ZEROS(nd, (npy_intp *)dims, type_num, fortran);
}

PyObject *__PyArray_Empty(int nd, npy_intp const *dims, PyArray_Descr *dtype,
                          int fortran) {
  return PyArray_Empty(nd, (npy_intp *)dims, dtype, fortran);
}

PyObject *__PyArray_EMPTY(int nd, npy_intp const *dims, int typenum,
                          int fortran) {
  return PyArray_EMPTY(nd, (npy_intp *)dims, typenum, fortran);
}

PyObject *__PyArray_Arange(double start, double stop, double step,
                           int typenum) {
  return PyArray_Arange(start, stop, step, typenum);
}

int __PyArray_SetBaseObject(PyObject *arr, PyObject *obj) {
  return PyArray_SetBaseObject((PyArrayObject *)arr, obj);
}

PyArray_Descr *__PyArray_DescrFromType(int typenum) {
  return PyArray_DescrFromType(typenum);
}

int __PyArray_INCREF(PyObject *arr) {
  return PyArray_INCREF((PyArrayObject *)arr);
}

int __PyArray_XDECREF(PyObject *arr) {
  return PyArray_XDECREF((PyArrayObject *)arr);
}

int __PyArray_Copy(PyObject *sarr, PyObject *darr) {
  void *sdata = PyArray_DATA((PyArrayObject *)sarr);
  void *ddata = PyArray_DATA((PyArrayObject *)darr);
  npy_intp snbytes = PyArray_NBYTES((PyArrayObject *)sarr);
  npy_intp dnbytes = PyArray_NBYTES((PyArrayObject *)sarr);
  if (snbytes != dnbytes) {
    return -1;
  } else {
    memcpy(ddata, sdata, snbytes);
  }

  return 0;
}

int __PyArray_IsFortran(PyObject *arr) {
  int flags = PyArray_FLAGS((PyArrayObject *)arr);
  return (flags & NPY_ARRAY_F_CONTIGUOUS) != 0;
}