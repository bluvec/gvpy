#ifndef GVPY_PYTHON_PY_H_
#define GVPY_PYTHON_PY_H_

#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

struct _object;
struct _typeobject;
struct _PyArray_Descr;
struct _ts;
typedef struct _object PyObject;
typedef struct _typeobject PyTypeObject;
typedef struct _ts PyThreadState;
typedef struct _PyArray_Descr PyArray_Descr;
typedef intptr_t npy_intp;
typedef ssize_t Py_ssize_t;
typedef int __PyGILState_STATE;

// error
PyObject *__PyErr_Occurred();
void __PyErr_PrintEx(int set_sys_last_vars);
void __PyErr_Print();
void __PyErr_Clear();

// lifecycle
void __Py_Initialize();
void __Py_InitializeEx(int initsigs);
int __Py_IsInitialized();
int __Py_FinalizeEx();

char *__Py_GetPath();
void __Py_SetPath(const char *path);

const char *__Py_GetVersion();

// pymem
void *__PyMem_Malloc(size_t n);
void *__PyMem_Realloc(void *p, size_t n);
void __PyMem_Free(void *p);

// pystate
int __PyGILState_Check();
__PyGILState_STATE __PyGILState_Ensure();
void __PyGILState_Release(__PyGILState_STATE s);

// ceval
int __PyEval_ThreadsInitialized();
void __PyEval_InitThreads();

PyThreadState *__PyEval_SaveThread();
void __PyEval_RestoreThread(PyThreadState *tstate);

// pythonrun
int __PyRun_SimpleString(const char *command);

// object
PyObject *__Py_None();
void __Py_CLEAR(PyObject *o);
void __Py_XINCREF(PyObject *o);
void __Py_XDECREF(PyObject *o);
Py_ssize_t __Py_REFCNT(PyObject *o);

PyTypeObject *__Py_TYPE(PyObject *o);

int __PyObject_IsInstance(PyObject *p, PyObject *c);

PyObject *__PyObject_Str(PyObject *o);

int __PyObject_HasAttrString(PyObject *o, const char *attr_name);
PyObject *__PyObject_GetAttrString(PyObject *o, const char *attr_name);
int __PyObject_SetAttrString(PyObject *o, const char *attr_name, PyObject *v);

int __PyCallable_Check(PyObject *o);
PyObject *__PyObject_Call(PyObject *p, PyObject *args, PyObject *kwargs);
PyObject *__PyObject_CallObject(PyObject *p, PyObject *args);

// import
PyObject *__PyImport_ImportModule(const char *name);
PyObject *__PyImport_ImportModuleLevel(const char *name, PyObject *globals,
                                       PyObject *locals, PyObject *fromlist,
                                       int level);

// sysmodule
PyObject *__PySys_GetObject(const char *name);
int __PySys_SetObject(const char *name, PyObject *v);

// boolobject
PyObject *__Py_True();
PyObject *__Py_False();
int __PyBool_Check(PyObject *o);
PyObject *__PyBool_FromLong(long v);

// longobject
int __PyLong_Check(PyObject *o);
long __PyLong_AsLong(PyObject *obj);
PyObject *__PyLong_FromLong(long v);

// floatobject
int __PyFloat_Check(PyObject *o);
double __PyFloat_AsDouble(PyObject *pyfloat);
PyObject *__PyFloat_FromDouble(double v);

// bytesobject
int __PyBytes_Check(PyObject *o);
PyObject *__PyBytes_FromString(const char *v);
PyObject *__PyBytes_FromStringAndSize(const char *v, Py_ssize_t len);
Py_ssize_t __PyBytes_Size(PyObject *o);
char *__PyBytes_AsString(PyObject *o);

// listobject
int __PyList_Check(PyObject *o);
PyObject *__PyList_New(Py_ssize_t len);
Py_ssize_t __PyList_Size(PyObject *list);
PyObject *__PyList_GetItem(PyObject *list, Py_ssize_t index);
int __PyList_SetItem(PyObject *list, Py_ssize_t index, PyObject *item);
int __PyList_Insert(PyObject *list, Py_ssize_t index, PyObject *item);
int __PyList_Append(PyObject *list, PyObject *item);
PyObject *__PyList_AsTuple(PyObject *list);

// dictobject
int __PyDict_Check(PyObject *o);
PyObject *__PyDict_New();
PyObject *__PyDict_GetItem(PyObject *p, PyObject *key);
PyObject *__PyDict_GetItemString(PyObject *p, const char *key);
int __PyDict_SetItem(PyObject *p, PyObject *key, PyObject *val);
int __PyDict_SetItemString(PyObject *p, const char *key, PyObject *val);
int __PyDict_DelItem(PyObject *p, PyObject *key);
int __PyDict_DelItemString(PyObject *p, const char *key);
int __PyDict_Contains(PyObject *p, PyObject *key);
void __PyDict_Clear(PyObject *p);
PyObject *__PyDict_Items(PyObject *p);
PyObject *__PyDict_Keys(PyObject *p);
PyObject *__PyDict_Values(PyObject *p);
Py_ssize_t __PyDict_Size(PyObject *p);
int __PyDict_Next(PyObject *p, Py_ssize_t *ppos, PyObject **pkey,
                  PyObject **pvalue);
int __PyDict_Merge(PyObject *a, PyObject *b, int override);
int __PyDict_Update(PyObject *a, PyObject *b);

// tupleobject
int __PyTuple_Check(PyObject *o);
PyObject *__PyTuple_New(Py_ssize_t len);
Py_ssize_t __PyTuple_Size(PyObject *p);
PyObject *__PyTuple_GetItem(PyObject *p, Py_ssize_t pos);
int __PyTuple_SetItem(PyObject *p, Py_ssize_t pos, PyObject *o);

// funcobject
int __PyFunction_Check(PyObject *o);

// unicodeobject
int __PyUnicode_Check(PyObject *o);
PyObject *__PyUnicode_FromString(const char *u);
const char *__PyUnicode_AsUTF8(PyObject *unicode);

// moduleobject
int __PyModule_Check(PyObject *o);

// classobject
int __PyClass_Check(PyObject *o);

// arrayobject
int __PyArray_import_array();
int __PyArray_Check(PyObject *o);
int __PyArray_NDIM(PyObject *arr);
int __PyArray_FLAGS(PyObject *arr);
int __PyArray_TYPE(PyObject *arr);
int __PyArray_SETITEM(PyObject *arr, void *itemptr, PyObject *obj);
void *__PyArray_DATA(PyObject *arr);
char *__PyArray_BYTES(PyObject *arr);
npy_intp *__PyArray_DIMS(PyObject *arr);
npy_intp *__PyArray_SHAPE(PyObject *arr);
npy_intp *__PyArray_STRIDES(PyObject *arr);
npy_intp __PyArray_DIM(PyObject *arr, int n);
npy_intp __PyArray_STRIDE(PyObject *arr, int n);
npy_intp __PyArray_ITEMSIZE(PyObject *arr);
npy_intp __PyArray_SIZE(PyObject *arr);
npy_intp __PyArray_Size(PyObject *obj);
npy_intp __PyArray_NBYTES(PyObject *arr);
PyObject *__PyArray_BASE(PyObject *arr);
PyArray_Descr *__PyArray_DESCR(PyObject *arr);
PyArray_Descr *__PyArray_DTYPE(PyObject *arr);
PyObject *__PyArray_GETITEM(PyObject *arr, void *itemptr);

// arrayobject: data access
void *__PyArray_GetPtr(PyObject *aobj, npy_intp *ind);
void *__PyArray_GETPTR1(PyObject *obj, npy_intp i);
void *__PyArray_GETPTR2(PyObject *obj, npy_intp i, npy_intp j);
void *__PyArray_GETPTR3(PyObject *obj, npy_intp i, npy_intp j, npy_intp k);
void *__PyArray_GETPTR4(PyObject *obj, npy_intp i, npy_intp j, npy_intp k,
                        npy_intp l);

// arrayobject: creating arrays
PyObject *__PyArray_SimpleNew(int nd, npy_intp *dims, int typenum);
PyObject *__PyArray_SimpleNewFromData(int nd, npy_intp const *dims, int typenum,
                                      void *data);
PyObject *__PyArray_Zeros(int nd, npy_intp const *dims, PyArray_Descr *dtype,
                          int fortran);
PyObject *__PyArray_ZEROS(int nd, npy_intp const *dims, int type_num,
                          int fortran);
PyObject *__PyArray_Empty(int nd, npy_intp const *dims, PyArray_Descr *dtype,
                          int fortran);
PyObject *__PyArray_EMPTY(int nd, npy_intp const *dims, int typenum,
                          int fortran);
PyObject *__PyArray_Arange(double start, double stop, double step, int typenum);
int __PyArray_SetBaseObject(PyObject *arr, PyObject *obj);

// arrayobject: data-type descriptors
PyArray_Descr *__PyArray_DescrFromType(int typenum);

// arrayobject: special functions for NPY_OBJECT
int __PyArray_INCREF(PyObject *arr);
int __PyArray_XDECREF(PyObject *arr);

// arrayobject: misc
int __PyArray_Copy(PyObject *sarr, PyObject *darr);
int __PyArray_IsFortran(PyObject *arr);

#ifdef __cplusplus
}
#endif

#endif  // GVPY_PYTHON_PY_H_
