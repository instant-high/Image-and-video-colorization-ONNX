import argparse

import platform
import os
import subprocess

import cv2
import numpy
from tqdm import tqdm
from skimage import img_as_ubyte

from colorizer.saturation import adjust_saturation

# options #
parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--audio', action='store_true', help='keep audio')
parser.add_argument('--mode', dest="mode", type=int, default=1, help='eccv16(0) or siggraph17(1)')
parser.add_argument('--fp16', action='store_true', help='whether to use fp16 model')
parser.add_argument("--render_factor", type=int, default=10, help=" - ")
parser.add_argument("--saturation", type=float, default=1.0, help="Adjust color saturation of result image ")

opt = parser.parse_args()

#
render_factor = opt.render_factor * 32
sat = opt.saturation
#

if opt.mode == 0:
    if not opt.fp16:
        from colorizer.colorizer import COLORIZER
        color = COLORIZER(model_path="colorizer/colorizer_eccv16.onnx", device="cuda")
    else:
        from colorizer.colorizer_fp16 import COLORIZER
        color = COLORIZER(model_path="colorizer/colorizer_eccv16_fp16.onnx", device="cuda")
	
if opt.mode == 1:
    if not opt.fp16:
        from colorizer.colorizer import COLORIZER
        color = COLORIZER(model_path="colorizer/colorizer_siggraph17.onnx", device="cuda")
    else:
        from colorizer.colorizer_fp16 import COLORIZER	  
        color = COLORIZER(model_path="colorizer/colorizer_siggraph17_fp16.onnx", device="cuda")


if os.path.exists('_temp.mp4'):
    os.remove('_temp.mp4')

from colorizer.saturation import adjust_saturation
sat = opt.saturation

video = cv2.VideoCapture(opt.input)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))       
fps = video.get(cv2.CAP_PROP_FPS)
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

output = cv2.VideoWriter(opt.output,cv2.VideoWriter_fourcc('m','p','4','v'), fps, (width,height))

if opt.audio:
    temp_video = '_temp.mp4'
    output = cv2.VideoWriter((temp_video),cv2.VideoWriter_fourcc('m','p','4','v'), fps, (width,height))

render_factor = 384
          		
for i in tqdm(range(length)):
    ret,img = video.read()
	
    result  = color.colorize(img, render_factor)
    result = adjust_saturation(result, sat)
  	
    output.write(img_as_ubyte(result))
    cv2.imshow("Colorized Video, press 'Esc' to stop",result)
    k = cv2.waitKey(1) 
    if k == 27:
        cv2.destroyAllWindows()
        output.release()
        break
		
cv2.destroyAllWindows()
output.release()

if opt.audio:
    print ("Writing Audio... ")
	
    command = 'ffmpeg.exe -y -vn -i ' + '"' + opt.input + '"' + ' -an -i ' + temp_video + ' -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest ' + '"' + opt.output + '"'
    subprocess.call(command, shell=platform.system() != 'Windows')
    #os.system('cls') 
    #input("Press Enter to exit...") 
    if os.path.exists(temp_video):
        os.remove(temp_video) 