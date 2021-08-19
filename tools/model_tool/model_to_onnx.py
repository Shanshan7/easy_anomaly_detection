#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from torch_onnx.torch_to_onnx import TorchConvertOnnx

class ModelConverter():

    def __init__(self, input_size=(352, 640)):
        self.converter = TorchConvertOnnx()
        self.input_size = input_size  # w * h

    def convert_process(self, net_config, weight_path, save_dir,
                        input_names=None, output_names=None):
        self.converter.set_input_names(input_names)
        self.converter.set_output_names(output_names)
        onnx_path = self.model_convert(net_config, weight_path, save_dir)
        return onnx_path

    def model_convert(self, net_config, weight_path, save_dir):
        data_channel = 3
        input_x = torch.randn(1, data_channel, self.input_size[1], self.input_size[0])
        self.converter.set_input(input_x)
        self.converter.set_save_dir(save_dir)
        save_onnx_path = self.converter.torch2onnx(net_config, weight_path)
        return save_onnx_path

    def base_model_convert(self, net_config, weight_path, save_dir):
        input_x = torch.randn(1, 3, self.input_size[1], self.input_size[0])
        self.converter.set_input(input_x)
        self.converter.set_save_dir(save_dir)
        save_onnx_path = self.converter.torch2onnx(net_config, weight_path)
        return save_onnx_path
