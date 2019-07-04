# Deep-Video-Composting

A python script for video composting .

It takes two videos as input and produces a single video composite as output. The person in the first video is segmented out and blended seamlessly into the second video (background) .It uses deeplab v3 for segmentation and a custom network (caffe) for color harmonization.Since the tensorflow model was trained on PASCAL VOC dataset, we can segment out any object belonging to those classes and combine them with the other video to produce a video composite.

Sample input and output videos along with link to pretrained modles are provided in respective folders.

N.B: Only the inference code is provides here, please refer the links in the acknowledgement section for training and other implementation details.

## Dependencies

* Python 2.7
* Tensorflow
* Caffe
* Opencv, PIL, Scikit-Image

## Prerequisites

* Download model files : https://drive.google.com/open?id=1XRxpVu2I70Pu0IchXXEcfq9GDQw8XeT5
* GPU with CUDA support

## Screenshot

![Screenshot](isketch_screenshot1.jpg)


## Versioning

Version 1.0

## Authors

Anil Sathyan

## Acknowledgments
* "https://averdones.github.io/real-time-semantic-image-segmentation-with-deeplab-in-tensorflow/
* "https://github.com/tensorflow/models/tree/master/research/deeplab"
* "https://github.com/wasidennis/DeepHarmonization"

