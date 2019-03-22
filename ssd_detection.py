#!/usr/bin/python
#-*-coding:utf-8-*-
#!/usr/bin/env python
# Run this demo in ./caffe_ssd/examples/
caffe_root="/root/caffe/"
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib qt
from google.protobuf import text_format
from caffe.proto import caffe_pb2
#from utils.timer import Timer
import scipy.io as sio
import scipy.misc as smisc
import caffe, os, sys, time
import argparse
import time #计时器



def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames



def demo(net, image_name, transformer, labelmap, image_dir):
    """Detect object classes in an image using pre-computed object proposals."""

    fig, ax = plt.subplots()

    # Load an image and detection
    # set net to batch size of 1

    #image = caffe.io.load_image('examples/images/fish-bike.jpg')
    image = caffe.io.load_image(os.path.join(image_dir, image_name))
    plt.imshow(image)

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['detection_out']

    # Parse the outputs and generate proposals
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6. Filter most of false positives.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.05]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)

    # these [xmin ymin xmax ymax] are relative coordinates, from [0,1]
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()  #前景框颜色从0到1设置21类

    # Display the value of intermediate variables
    #print top_conf
    #print det_label[top_indices]
    #print top_label_indices
    #print top_labels
    #print top_xmin
    plt.imshow(image)
    #currentAxis = plt.gca()
    #print 'image shape = ' + str(image.shape)

    # Output the valid proposals
    for i in xrange(top_conf.shape[0]):
        # these [xmin ymin xmax ymax] are abosolute coordinates, from [0, image_width, 0, image_height]
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))

        # Adding some relative padding area in this section with some boundary inspectations
        # padding = compute reasonable padding area
        # xmin = xmin - padding
        # ymin = ymin - padding
        # xmax = xmax + padding
        # ymax = ymax + padding

        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%s: %.2f'%(label_name, score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        ax.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    plt.draw()
    plt.savefig("gen:/"+image_name)  #保存图片到路径gen
    plt.show()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SSD demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=2, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    args = parser.parse_args()

    return args



if __name__ == '__main__':

    args = parse_args()
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)

    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # Make sure that caffe is on the python path:
    caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
    os.chdir(caffe_root)
    sys.path.insert(0, 'python')


    # load PASCAL VOC labels
    labelmap_file = '/root/car/labelmap_voc.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)


    # load the trained model
    model_def = '/root/car/deploy.prototxt'
    model_weights = 'root/car/VGG_VOC21_SSD_500x500_iter_20000.caffemodel'

    net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    image_resize = 500
    net.blobs['data'].reshape(1,3,image_resize,image_resize)

	# Folder of images
    image_dir = '/root/car/test_image/';

	# List of images
    im_list = '/root/car/jpglist.txt';

    with open(im_list, 'r') as fid_im_list:
        im_names = fid_im_list.readlines()
    fid_im_list.close()

    #timerO = Timer()
    #timerO.tic()
    time_start = time.time() #计时开始
    count = 0;
    for line in im_names:
        count = count + 1;
        temp = line.split('\n')
        im_name = temp[0]
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        #print 'Demo for data/demo/{}'.format(im_name)
        print 'Processing image {} / {}'.format(count, len(im_names))
        demo(net, im_name, transformer, labelmap, image_dir)

    #timerO.toc()
    time_end = time.time() #测试完成计时结束
    time_cost = time_end - time_start;
    print ('Detection took {:.3f}s for '
           '{:d} images').format(time_cost, count)

    plt.show()


