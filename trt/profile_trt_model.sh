#! /bin/bash

sudo /usr/local/cuda-10.0/bin/nvprof -o int8.nvvp -f /usr/src/tensorrt/bin/trtexec --loadEngine=int8.engine --iterations=20 --batch=1

sudo /usr/local/cuda-10.0/bin/nvprof -o fp16.nvvp -f /usr/src/tensorrt/bin/trtexec --loadEngine=fp16.engine --iterations=20 --batch=1

sudo /usr/local/cuda-10.0/bin/nvprof -o fp32.nvvp -f /usr/src/tensorrt/bin/trtexec --loadEngine=fp32.engine --iterations=20 --batch=1
