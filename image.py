import cv2
import numpy as np
import argparse
import onnxruntime as ort

from colorizer.saturation import adjust_saturation

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--mode', dest="mode", type=int, default=1, help='eccv16(0) or siggraph17(1)')
parser.add_argument('--render_factor', type=int, default=10, help=" - ")
parser.add_argument('--saturation', type=float, default=1.0, help="Adjust color saturation of result image ")
opt = parser.parse_args()

#
render_factor = opt.render_factor * 32
sat = opt.saturation
#

if opt.mode == 0:
    from colorizer.colorizer import COLORIZER
    color = COLORIZER(model_path="colorizer/colorizer_eccv16.onnx", device="cuda")
    
if opt.mode == 1:
    from colorizer.colorizer import COLORIZER
    color = COLORIZER(model_path="colorizer/colorizer_siggraph17.onnx", device="cuda")

source = cv2.imread(opt.input)

result  = color.colorize(source, render_factor)
result = adjust_saturation(result, sat)

cv2.imshow("Colorized image",result)
cv2.waitKey()
cv2.imwrite(opt.output, result)


