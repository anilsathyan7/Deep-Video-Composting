import cv2
import numpy as np
import sys
import os

from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter

import cv2
import numpy as np
import os
from random import shuffle
import tensorflow as tf
import caffe
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.transform import resize
import time
import matplotlib.pyplot as plt

import collections
import io
from PIL import Image

sys.path.append('utils')

class DeepLabModel():
    """Class to load deeplab model and run inference."""
    
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    

    def __init__(self):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        
        graph_def = 'models/frozen_inference_graph_dm05.pb'
        with gfile.FastGFile(graph_def, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
      
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():      
            tf.import_graph_def(graph_def, name='')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)
            
    def run(self, image):
        """Runs inference on a single image.
        
        Args:
            image: A PIL.Image object, raw input image.
            
        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return seg_map

model = DeepLabModel()

def imresize(image):
    INPUT_SIZE = 513
    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    return resized_image


# caffe harmonizaton

# set up caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# load net
net = caffe.Net('models/deploy_512.prototxt', 'models/harmonize_iter_200000.caffemodel', caffe.TEST)

size = np.array([512,512])

# Open Video

vidcap1 = cv2.VideoCapture('videos/oldman_left_pal.mp4')
vidcap2 = cv2.VideoCapture('videos/autumn_right_pal.mp4')

outvid = cv2.VideoWriter('outvid.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1024,512))
## Get video fps

fps1 = vidcap1.get(cv2.CAP_PROP_FPS)
print('Frames per second of video 1 : {0}'.format(fps1))

fps1 = vidcap2.get(cv2.CAP_PROP_FPS)
print('Frames per second of video 2: {0}'.format(fps1))

# Segmentation Mobilenet

i=0
while True:
    
    start=time.time()

    (success1, image1) = vidcap1.read()
    (success2, image2) = vidcap2.read()

    if not (success1 and success2):
        continue

    else: 
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

       
        image1 = imresize(Image.fromarray(image1))
        image2 = imresize(Image.fromarray(image2))

    
        model_start_time=time.time()

        seg_map_left = model.run(image1)
        seg_map_left[seg_map_left>0]=255
        floatmap = seg_map_left.astype(np.float32)/255.0
        blurmap = gaussian_filter(floatmap, sigma=3)
        cv2.imwrite("blurseg.png",blurmap*255)
       
        model_ex_time=time.time()-model_start_time
        print(model_ex_time)
    

        resized_im_left=np.float32(image1)
        resized_im_right=np.float32(image2)

   
        print( resized_im_left.shape)
        print( resized_im_right.shape)

        resized_im_right=resized_im_right.copy()
        blurmap = blurmap[..., np.newaxis]
        
        frame = (resized_im_left * blurmap) + (resized_im_right* (1 - blurmap))
     
        frame=np.uint8(frame)[..., ::-1]
        cv2.imwrite("before.png",frame)
        msk = np.uint8(blurmap*255)
        dim = (512, 512)
        msk = cv2.resize(msk,dim)

        end=time.time()
        elapsed=end-start
        print(elapsed)        
        

        #Blending ....
        im_ori = Image.open('before.png')
	

        im = im_ori.resize(size, Image.BICUBIC)
        im = np.array(im, dtype=np.float32)
        if im.shape[2] == 4:
            im = im[:,:,0:3]

        im = im[:,:,::-1]
        raw = np.uint8(im)
        im -= np.array((104.00699, 116.66877, 122.67892))
        im = im.transpose((2,0,1))
        mask = Image.open('blurseg.png')
                                                                                                                                                
        mask = mask.resize(size, Image.BICUBIC)
        mask = np.array(mask, dtype=np.float32)
        if len(mask.shape) == 3:
            mask = mask[:,:,0]

        mask -= 128.0
        mask = mask[np.newaxis, ...]

	# shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *im.shape)
        net.blobs['data'].data[...] = im

        net.blobs['mask'].reshape(1, *mask.shape)
        net.blobs['mask'].data[...] = mask

	# run net for prediction
        net.forward()
        out = net.blobs['output-h'].data[0]
        out = out.transpose((1,2,0))
        out += np.array((104.00699, 116.66877, 122.67892))
        out = out[:,:,::-1]
  
        neg_idx = out < 0.0
        out[neg_idx] = 0.0
        pos_idx = out > 255.0
        out[pos_idx] = 255.0

  	# save result
        result = out.astype(np.uint8)
        frame = result[..., ::-1]
                                                                                                                                                    
        cv2.imwrite("after.png",frame)
        cv2.imshow('frame', frame)
        
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            vidcap1.release()
            vidcap2.release()
            cv2.destroyAllWindows()
            break 

        # Save the video
        result_all = np.concatenate((raw, frame), axis = 1)
        outvid.write(result_all)
