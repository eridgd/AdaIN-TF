# AdaIN-TF

This is a TensorFlow/Keras implementation of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868).

## Requirements

* Python 3.x (2.7 may work but is untested)
* tensorflow 1.2.1+
* keras 2.0.x
* torchfile 

Optionally:
* OpenCV with contrib modules (for webcam.py)
  * MacOS install http://www.pyimagesearch.com/2016/12/05/macos-install-opencv-3-and-python-3-5/
  * Linux install http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/
* ffmpeg (for video stylization)

## Running a pre-trained model

## Training

1. Download [MSCOCO images](http://mscoco.org/dataset/#download) and [Wikiart images](https://www.kaggle.com/c/painter-by-numbers).



## Acknowledgments

Many thanks to the author Xun Huang for the excellent [original Torch implementation](https://github.com/xunhuang1995/AdaIN-style) that saved me countless hours of frustation. I also drew inspiration from Jon Rei's [TF implementation for the .t7 pre-trained decoder](https://github.com/jonrei/tf-AdaIN) and borrowed the clean TF code for the AdaIN transform.

# TODO:
* Docs
* Pre-trained model
* Pre-compute style encoding means/stds
* CORAL for preserving colors
* Video processing
* Model freezing
* Webcam style window threading
* Keras VGG