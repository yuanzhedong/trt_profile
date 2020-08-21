import os
import sys
import time

import tensorrt as trt
from PIL import Image
import numpy as np
import torch
import utils.calibrator as calibrator
import argparse
import uff

TRT_LOGGER = trt.Logger()

class ModelData(object):
    def __init__(self, pb_file_path, input_name, trt_input_shape, output_name, fp16_mode, int8_mode, trt_engine_path, cali_data_dir):
        self.pb_file_path = pb_file_path
        self.input_name = input_name
        self.input_shape = trt_input_shape
        self.output_name = output_name
        self.fp16_mode = fp16_mode
        self.int8_mode = int8_mode
        self.trt_engine_path = trt_engine_path
        self.cali_data_dir = cali_data_dir

def GiB(val):
    return val * 1 << 30

def build_engine(model_data):
    uff_model = uff.from_tensorflow_frozen_model(model_data.pb_file_path)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = GiB(6)
        parser.register_input(model_data.input_name, model_data.input_shape)
        parser.register_output(model_data.output_name)
        parser.parse_buffer(uff_model, network)
        if model_data.int8_mode:
            builder.int8_mode = True
            builder.int8_calibrator = calibrator.EntropyCalibrator(data_dir=model_data.cali_data_dir, cache_file='int8calicache')
        else:
            if model_data.fp16_mode:
                builder.fp16_mode = True
        return builder.build_cuda_engine(network)

def save_engine(engine, engine_dest_path):
    print('Engine:', engine)
    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)

def arg_parse(): 
        parser = argparse.ArgumentParser(description='create_trt_engine')

        parser.add_argument("--pb_file", dest = 'pb_file', help = "frozen pb file path", default = "/media/nvidia/ssd/autopilot/scripts/int8/frozen_model.pb", type = str)
        parser.add_argument("--input_name", dest = 'input_name', help = "input node name", default = "data", type = str)
        parser.add_argument("--output_name", dest = 'output_name', help = "output node name", default = "sigmoid/Sigmoid", type = str)
        parser.add_argument("--fp16", action='store_true')
        parser.add_argument("--int8", action='store_true')
        parser.add_argument("--fp32", action='store_true')
        parser.add_argument("--cali_data_dir", dest = "cali_data_dir", help = "calibration dataset", default = "/media/nvidia/ssd/autopilot/data/output/image/00044bcc0a9f/smooth/")
        return parser.parse_args()

def main():
        args = arg_parse()
                               
        pb_file = args.pb_file
        input_name = args.input_name
        output_name = args.output_name
        cali_data_dir = args.cali_data_dir

        if args.fp16:
            model_data = ModelData(pb_file,
                                   input_name,
                                   (3, 512, 512),
                                   output_name,
                                   True,
                                   False,
                                   "fp16.engine",
                                   cali_data_dir)
            engine = build_engine(model_data)
            save_engine(engine, model_data.trt_engine_path)
            
        if args.int8:
            model_data = ModelData(pb_file,
                                   input_name,
                                   (3, 512, 512),
                                   output_name,
                                   False,
                                   True,
                                   "int8.engine",
                                   cali_data_dir)
            engine = build_engine(model_data)
            save_engine(engine, model_data.trt_engine_path)

        if args.fp32:
            model_data = ModelData(pb_file,
                                   input_name,
                                   (3, 512, 512),
                                   output_name,
                                   False,
                                   False,
                                   "fp32.engine",
                                   cali_data_dir)
            engine = build_engine(model_data)
            save_engine(engine, model_data.trt_engine_path)

if __name__ == '__main__':
        main()

