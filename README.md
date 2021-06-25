# Gvpy
[![gvpy](https://raw.githubusercontent.com/bluvec/gvpy/readme/res/gvpylogo.png)](https://github.com/bluvec/gvpy)

[![Go Reference](https://pkg.go.dev/badge/github.com/bluvec/gvpy.svg)](https://pkg.go.dev/github.com/bluvec/gvpy)
[![Go Report Card](https://goreportcard.com/badge/github.com/bluvec/gvpy)](https://goreportcard.com/report/github.com/bluvec/gvpy)
[![License](https://img.shields.io/github/license/bluvec/gvpy)](https://raw.githubusercontent.com/bluvec/gvpy/readme/LICENSE)

**Gvpy** is short for **G**o **V**isits **PY**thon.

Gvpy provides scalable high-level APIs to call python codes from golang. Compared to [go-python3]([go-python3](https://github.com/DataDog/go-python3)) which provides the low-level golang bindings for the C-API of cpython, gvpy is easier to use and memory safer due to the help of golang garbage collector. Moreover, the _numpy_ package is also supported by gvpy.

**NOTES**
* Gvpy only supports python3.
* Gvpy is currently only tested in Linux systems. MacOSX and Windows supported will be added in the near future.
* Gvpy is only tested with python >=3.6.
* Gvpy is still under heavy development. Every API is subject to change before stable.

## Installation
### Before go get
Gvpy needs the `cpython` library and the popular python package `numpy`. Before `go get`, make sure they have been installed.
* Ubuntu
  ```bash
  $ sudo apt update
  $ sudo apt install pkg-config
  $ sudo apt install python3 python3-dev python3-numpy
  ```
* Conda
  ```bash
  $ sudo apt install pkg-config
  $ conda create -n py38 python=3.8
  $ conda activate py38
  $ conda install -c conda-forge numpy
  $ export PKG_CONFIG_PATH=${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}
  $ ln -s ${CONDA_PREFIX}/lib/python3.8/site-packages/numpy/core/include/numpy ${CONDA_PREFIX}/include/python3.8/
  ```

  It also works to help gvpy find python and numpy libraries by setting environment variables `CGO_CFLAGS` and `CGO_LDFLAGS` as follows:
  ```bash
  $ export CGO_CFLAGS="-I${CONDA_PREFIX}/include/python3.8 -I${CONDA_PREFIX}/lib/python3.8/site-packages/numpy/core/include/numpy"
  $ export CGO_LDFLAGS="-L${CONDA_PREFIX}/lib -lpython3.8"
  ```

### Go get
```bash
$ go get -u github.com/bluvec/gvpy
```

By default, gvpy only supports python >= 3.8. To get gvpy with python 3.5, 3.6 and 3.7, the build tag is needed.
```bash
$ go get -u -tags py35 github.com/bluvec/gvpy # get gvpy with python3.5
$ go get -u -tags py36 github.com/bluvec/gvpy # get gvpy with python3.6
$ go get -u -tags py37 github.com/bluvec/gvpy # get gvpy with python3.7
```

## Examples

## Notes on using lowlevel APIs
### Global Interpreter Lock (GIL)

### Golang thread scheduler

