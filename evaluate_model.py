#!/usr/bin/env python
"""
Classify an image using individual model files

Use this script as an example to build your own tool
"""

import argparse
import os
import time

import PIL.Image
import numpy as np
import scipy.misc
from google.protobuf import text_format
from collections import Counter

os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output
import caffe
from caffe.proto import caffe_pb2

def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net

    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)

def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer

    Arguments:
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        with open(mean_file,'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t

def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk

    Returns an np.ndarray (channels x width x height)

    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension

    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

def forward_pass(images, net, transformer, batch_size=1):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)

    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer

    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:,:,np.newaxis])
        else:
            caffe_images.append(image)

    caffe_images = np.array(caffe_images)

    dims = transformer.inputs['data'][1:]

    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        output = net.forward()[net.outputs[-1]]
        if scores is None:
            scores = output
        else:
            scores = np.vstack((scores, output))
        print 'Processed %s/%s images ...' % (len(scores), len(caffe_images))

    return scores

def read_labels(labels_file):
    """
    Returns a list of strings

    Arguments:
    labels_file -- path to a .txt file
    """
    if not labels_file:
        print 'WARNING: No labels file provided. Results will be difficult to interpret.'
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels

def get_class_label(filename):
	class_name = os.path.split(os.path.dirname(os.path.abspath(filename)))[1]
	return class_name

def classify(caffemodel, deploy_file, image_files,
        mean_file=None, labels_file=None, use_gpu=True):
    """
    Classify some images against a Caffe model and print the results

    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images

    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    # Load the model and images
    net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
    images = [load_image(image_file, height, width, mode) for image_file in image_files]
    labels = read_labels(labels_file)

    # Classify the image
    classify_start_time = time.time()
    scores = forward_pass(images, net, transformer,batch_size=100)
    print 'Classification took %s seconds.' % (time.time() - classify_start_time,)

    ### Process the results
    fp = Counter()
    fn = Counter()
    tp = Counter()
    tn = Counter()

    indices = (-scores).argsort()[:, :1] # take top 1 result
    classifications = []
    print labels
    for image_index, index_list in enumerate(indices):
        result = []
        for i in index_list:
            # 'i' is a category in labels and also an index into scores
            if labels is None:
                label = 'Class #%s' % i
            else:
                label = labels[i]
            result.append((label, round(100.0*scores[image_index, i],4)))
        classifications.append(result)

    for index, classification in enumerate(classifications):
	true_class = get_class_label(image_files[index])
	if (true_class == classification[0][0]): 
	    tp[true_class] += 1
	else:
	    fp[true_class] += 1

    return tp[true_class], fp[true_class]

if __name__ == '__main__':
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Classification example - DIGITS')

    ### Positional arguments

    parser.add_argument('caffemodel',   help='Path to a .caffemodel')
    parser.add_argument('deploy_file',  help='Path to the deploy file')
    parser.add_argument('image',        help='Path to an image')

    ### Optional arguments

    parser.add_argument('-m', '--mean',
            help='Path to a mean file (*.npy)')
    parser.add_argument('-l', '--labels',
            help='Path to a labels file')
    parser.add_argument('--nogpu',
            action='store_true',
            help="Don't use the GPU")

    args = vars(parser.parse_args())

    fp = Counter()
    fn = Counter()
    tp = Counter()
    tn = Counter()
    true_class = ''
    image_folder = [args['image']]
    print image_folder
    image_folders = os.listdir(image_folder[0])
    for folders in image_folders:
	folder = os.path.join(image_folder[0],folders)
	for (dir, _, files) in os.walk(folder):
	    image_files = []
	    for image_file in files:
	        if image_file.endswith(('.png', '.JPG', '.jpg')):
		    image_files.append(os.path.join(dir,image_file))
            tp[folders], fp[folders] = classify(args['caffemodel'], args['deploy_file'], image_files,
                args['mean'], args['labels'], not args['nogpu'])

	    print tp, fp, true_class
    print "Scores: "
    total_num = 0
    total_acc = 0.0
    total_prec = 0.0
    total_recall = 0.0
    labels = read_labels(args['labels'])
    for true_class in labels:
	tp_val = tp[true_class]
	fp_val = fp[true_class]
	tn_val = tn[true_class]
	fn_val = fn[true_class]

	total = tp_val + fp_val + tn_val + fn_val
	total_num += total
	total_acc += (tp_val + tn_val)
	if total == 0:
		print "  True label %s -> # test points = %d" % (true_class, total)
		tt = 1
	else:
		accuracy = float(tp_val + tn_val) / (tp_val + fp_val + tn_val + fn_val)
		
		precision = -1.0
		if tp_val + fp_val > 0:
			precision = float(tp_val) / (tp_val + fp_val)

		if precision == -1:
			precision = 0
		
		recall = -1.0
		if tp_val + fn_val > 0:
			recall = float(tp_val) / (tp_val + fn_val)

		if recall == -1:
			recall = 0

		total_prec += precision
		total_recall += recall

		print "  True label %s -> # test points = %d; TP = %d; TN = %d; FP = %d; FN = %d; acc = %f; prec = %f; recall = %f" % \
			(true_class, total, tp_val, tn_val, fp_val, fn_val, accuracy, precision, recall)

    print "Total # of test points = %d \n" % total_num
    print "Average accuracy = " + str(total_acc / float(total_num)) + "\n"

    print 'Script took %s seconds.' % (time.time() - script_start_time,)

