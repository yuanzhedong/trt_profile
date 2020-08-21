#! /bin/bash

TRTEXEC=/usr/src/tensorrt/bin/trtexec
TRTEXEC=/home/landingai/TensorRT-5.1.5.0/bin/trtexec

NUM_ITER=100

/usr/local/cuda-10.0/bin/nvprof -o int8.nvvp -f $TRTEXEC --loadEngine=int8.engine --iterations=$NUM_ITER --batch=1

/usr/local/cuda-10.0/bin/nvprof -o fp16.nvvp -f $TRTEXEC --loadEngine=fp16.engine --iterations=$NUM_ITER --batch=1

/usr/local/cuda-10.0/bin/nvprof -o fp32.nvvp -f $TRTEXEC --loadEngine=fp32.engine --iterations=$NUM_ITER --batch=1
