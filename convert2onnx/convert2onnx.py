'''
https://github.com/richzhang/colorization/tree/master
run this script in the root of the original repo
'''

import argparse
import cv2
from colorizers import *

import torch
import torch.onnx

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
opt = parser.parse_args()

device = 'cpu'

colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
	device = 'cuda'
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()


input_names = [ "input" ]
output_names = [ "output" ]
# Generate a random dummy input with dynamic axes
dummy_input_shape = (1, 1, 256, 256)
dummy_input = torch.randn(dummy_input_shape, device=device)

# Export colorizer_eccv16 model to ONNX
torch.onnx.export(
    colorizer_eccv16,
    dummy_input,
    "colorizer_eccv16.onnx",
    export_params=True,
    do_constant_folding=True,
    verbose=True,
    opset_version=11,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}
)

# Export colorizer_siggraph17 model to ONNX
torch.onnx.export(
    colorizer_siggraph17,
    dummy_input,
    "colorizer_siggraph17.onnx",
    export_params=True,
    do_constant_folding=True,
    verbose=True,
    opset_version=11,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}
)

