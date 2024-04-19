# Image-and-video-colorization
Simple image and video colorization using onnx converted ECCV16 or SIGGRAPH17 models.

Easy to install. Can be run on CPU or nVidia GPU

ffmpeg for video colorzation required.

Added floating point 16 model for faster inference.

.

For inference run:

Image: python image.py --source_image "image.jpg"

Video: python video.py --source "video.mp4" --result "video_colorized.mp4" --audio

optional parameters:

--mode 0 or 1 (eccv16 or siggraph17) default 1

--fp16

--render_factor 10 (default=10)

--saturation 1 (0.1 - 2.0)

.

Original repository:

https://github.com/richzhang/colorization

@inproceedings{zhang2016colorful,
  title={Colorful Image Colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={ECCV},
  year={2016}
}

@article{zhang2017real,
  title={Real-Time User-Guided Image Colorization with Learned Deep Priors},
  author={Zhang, Richard and Zhu, Jun-Yan and Isola, Phillip and Geng, Xinyang and Lin, Angela S and Yu, Tianhe and Efros, Alexei A},
  journal={ACM Transactions on Graphics (TOG)},
  volume={9},
  number={4},
  year={2017},
  publisher={ACM}
}
