#! /bin/bash

make docker_push_cpu_mlperf

while [ $? -ne 0 ]; do
  make docker_push_cpu_mlperf
done
