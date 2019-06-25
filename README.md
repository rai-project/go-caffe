# go-caffe

[![Build Status](https://dev.azure.com/dakkak/rai/_apis/build/status/rai-project.go-caffe)](https://dev.azure.com/dakkak/rai/_build/latest?definitionId=9)
[![Build Status](https://travis-ci.org/rai-project/go-caffe.svg?branch=master)](https://travis-ci.org/rai-project/go-caffe)[![Go Report Card](https://goreportcard.com/badge/github.com/rai-project/go-caffe)](https://goreportcard.com/report/github.com/rai-project/go-caffe)[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![](https://images.microbadger.com/badges/version/carml/go-caffe:ppc64le-gpu-latest.svg)](https://microbadger.com/images/carml/go-caffe:ppc64le-gpu-latest> 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe:ppc64le-cpu-latest.svg)](https://microbadger.com/images/carml/go-caffe:ppc64le-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe:amd64-cpu-latest.svg)](https://microbadger.com/images/carml/go-caffe:amd64-cpu-latest 'Get your own version badge on microbadger.com') [![](https://images.microbadger.com/badges/version/carml/go-caffe:amd64-gpu-latest.svg)](https://microbadger.com/images/carml/go-caffe:amd64-gpu-latest 'Get your own version badge on microbadger.com')

Go binding for Caffe C predict API.
This is used by the [Caffe agent](https://github.com/rai-project/caffe) in [MLModelScope](mlmodelscope.org) to perform model inference in Go.

## Installation

Download and install go-caffe:

```
go get -v github.com/rai-project/go-caffe
```

The repo requires Caffe and other Go packages.

### Caffe C Library

The Caffe C library is expected to be under `/opt/caffe`.

Please refer to [scripts](scripts) or the `LIBRARY INSTALLATION` section in the [dockefiles](dockerfiles) to install caffe on your system. OpenBLAS is used.

To install Caffe on your system, you can follow the [Caffe documentation](https://caffe.berkeleyvision.org/installation.html), or refer to our [scripts](scripts) or the `LIBRARY INSTALLATION` section in the [dockefiles](dockerfiles). OpenBLAS is used in our default build.

- The default blas is OpenBLAS.
  The default OpenBLAS path for mac os is `/usr/local/opt/openblas` if installed throught homebrew (openblas is keg-only, which means it was not symlinked into /usr/local, because macOS provides BLAS and LAPACK in the Accelerate framework).

- The default caffe installation path is `/opt/caffe` for linux, darwin and ppc64le without powerai; `/opt/DL/caffe` for ppc64le with powerai.

- The default CUDA path is `/usr/local/cuda`

See [lib.go](lib.go) for details.

If you get an error about not being able to write to `/opt` then perform the following

```
sudo mkdir -p /opt/caffe
sudo chown -R `whoami` /opt/caffe
```

If you are using Caffe docker images or other libary paths, change CGO_CFLAGS, CGO_CXXFLAGS and CGO_LDFLAGS enviroment variables. Refer to [Using cgo with the go command](https://golang.org/cmd/cgo/#hdr-Using_cgo_with_the_go_command).

For example,

```
    export CGO_CFLAGS="${CGO_CFLAGS} -I/tmp/caffe/include"
    export CGO_CXXFLAGS="${CGO_CXXFLAGS} -I/tmp/caffe/include"
    export CGO_LDFLAGS="${CGO_LDFLAGS} -L/tmp/caffe/lib"
```

### Go packages

You can install the dependency through `go get`.

```
cd $GOPATH/src/github.com/rai-project/tensorflow
go get -u -v ./...
```

Or use [Dep](https://github.com/golang/dep).

```
dep ensure -v
```

This installs the dependency in `vendor/`.

### Configure Environmental Variables

Configure the linker environmental variables since the Caffe C library is under a non-system directory. Place the following in either your `~/.bashrc` or `~/.zshrc` file

Linux
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/caffe/lib
export LD_LIBRARY_PATH=/opt/caffe/lib:$DYLD_LIBRARY_PATH

```

macOS
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/caffe/lib
export DYLD_LIBRARY_PATH=/opt/caffe/lib:$DYLD_LIBRARY_PATH
```

## Check the Build

Run `go build` in to check the dependences installation and library paths set-up.
On linux, the default is to use GPU, if you don't have a GPU, do `go build -tags nogpu` instead of `go build`.

**_Note_**: The CGO interface passes go pointers to the C API. This is an error by the CGO runtime. Disable the error by placing

```
export GODEBUG=cgocheck=0
```

in your `~/.bashrc` or `~/.zshrc` file and then run either `source ~/.bashrc` or `source ~/.zshrc`

## Examples

Examples of using the Go MXNet binding to do model inference are under [examples](examples).

### batch_mlmodelscope

This example shows how to use the MLModelScope tracer to profile the inference.

Refer to [Set up the external services](https://docs.mlmodelscope.org/installation/source/external_services/) to start the tracer.

Then run the example by

```
  cd example/batch_mlmodelscope
  go build
  ./batch
```

Now you can go to `localhost:16686` to look at the trace of that inference.

### batch_nvprof

This example shows how to use nvprof to profile the inference. You need GPU and CUDA to run this example.

```
  cd example/batch_nvprof
  go build
  nvprof --profile-from-start off ./batch_nvprof
```

Refer to [Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html) for using nvprof.


## Issues

- Install Caffe with CUDA 10.0, see https://github.com/clab/dynet/issues/1457.
