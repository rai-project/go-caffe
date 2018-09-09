# GO Bindings for Caffe Prediction [![Build Status](https://travis-ci.org/rai-project/go-caffe.svg?branch=master)](https://travis-ci.org/rai-project/go-caffe)

## Caffe Installation

The go bindings need a Caffe installation and to include caffe path in [lib.go](lib.go).

Please refer to the `LIBRARY INSTALLATION` section in the [dockefiles](dockerfiles) to install caffe on your system.

The default caffe installation path is `/opt/caffe` for linux, darwin and ppc64le w/o powerai; `/opt/DL/caffe` for ppc64le w/ powerai. See [lib.go](lib.go) for details.

## CUDA Installation

The default CUDA path is `/usr/local/cuda`
