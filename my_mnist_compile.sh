#!/bin/bash
ARCH=./arch.json
COMPILE_KV260=./outputs_kv260

vai_c_tensorflow2 \
    --model      ./quantized_model.h5 \
    --arch       ./arch.json \
    --output_dir ./outputs/ \
    --net_name   my_mnist_test
