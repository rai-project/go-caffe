# GO Bindings for Caffe Prediction [![Build Status](https://travis-ci.org/rai-project/go-caffe.svg?branch=master)](https://travis-ci.org/rai-project/go-caffe)

## Caffe Installation

Please refer to the `LIBRARY INSTALLATION` section in the [dockefiles](dockerfiles) to install caffe on your system. OpenBLAS is used.

- The default caffe installation path is `/opt/caffe` for linux, darwin and ppc64le w/o powerai; `/opt/DL/caffe` for ppc64le w/ powerai.

- The default OpenBLAS path for mac os is `/usr/local/opt/openblas` if installed throught homebrew (openblas is keg-only, which means it was not symlinked into /usr/local, because macOS provides BLAS and LAPACK in the Accelerate framework).

- The default CUDA path is `/usr/local/cuda`

See [lib.go](lib.go) for details.

To use other library paths, change CGO_CFLAGS, CGO_CXXFLAGS and CGO_LDFLAGS enviroment variables.

For example,

```
    export CGO_CFLAGS="${CGO_CFLAGS} -I /usr/local/cuda-9.2/include -I/usr/local/cuda-9.2/nvvm/include -I /usr/local/cuda-9.2/extras/CUPTI/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include/crt"
    export CGO_CXXFLAGS="${CGO_CXXFLAGS} -I /usr/local/cuda-9.2/include -I/usr/local/cuda-9.2/nvvm/include -I /usr/local/cuda-9.2/extras/CUPTI/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include -I /usr/local/cuda-9.2/targets/x86_64-linux/include/crt"
    export CGO_LDFLAGS="${CGO_LDFLAGS} -L /usr/local/nvidia/lib64 -L /usr/local/cuda-9.2/nvvm/lib64 -L /usr/local/cuda-9.2/lib64 -L /usr/local/cuda-9.2/lib64/stubs -L /usr/local/cuda-9.2/targets/x86_64-linux/lib/stubs/ -L /usr/local/cuda-9.2/lib64/stubs -L /usr/local/cuda-9.2/extras/CUPTI/lib64"
```
