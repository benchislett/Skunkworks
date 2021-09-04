# <-- Imports --> #

import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
import scipy
from scipy import misc as scimisc, io as sio, signal
from scipy.ndimage.filters import convolve
import os
import time
import random
import re
import shutil
import skimage
from math import floor
from skimage import data, img_as_float
from skimage.measure import compare_ssim

# <-- Constants & Variables --> #

width = 32
height = 32
channels = 3

in_shape = [None, width*height*channels]
out_shape = [None, width*height*channels]

batch_size = 64
epochs = 64

target_bits = 4608
target_filters = 72

# <-- Pulse Signal Noise Ratio (PSNR) --> #


def PSNR(x, y):
    mse = ((x - y) ** 2).mean()
    return np.log10(mse)*10.0

# <-- Tensorflow Visualization Initialization --> #


def summary(x):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(x)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(x - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(x))
        tf.summary.scalar('min', tf.reduce_min(x))
        tf.summary.histogram('histogram', x)

# <-- Color Space Conversion --> #
# - Disclaimer: Shamelessly stolen from stack overflow - #


def YCC2RGB(x):
    xform = np.array(
        [[1.0, 0.0, 1.402], [1.0, -0.34414, -0.71414], [1.0, 1.772, 0.0]])
    rgb = x.astype(np.float32)
    rgb[:, :, [1, 2]] -= 128.0
    rgb = rgb.dot(xform.T).astype(np.float32)
    return np.clip(rgb, 0.0, 255.0).astype(np.float32)


def RGB2YCC(x):
    xform = np.array(
        [[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]])
    ycc = x.astype(np.float32).dot(xform.T).astype(np.float32)
    ycc[:, :, [1, 2]] += 128.0
    return np.clip(ycc, 0.0, 255.0).astype(np.float32)

# <-- Data Set --> #

#from tensorflow.examples.tutorials.mnist import input_data
#data = input_data.read_data_sets('MNIST_data', one_hot=True)


train_loc = './train_32x32.mat'
test_loc = './test_32x32.mat'

train_data = sio.loadmat(train_loc)
test_data = sio.loadmat(test_loc)

train_x = train_data['X'] / 255.0
train_y = train_data['y']

test_x = test_data['X'] / 255.0
test_y = test_data['y']


def shuffle_train():
    global train_samples
    global train_y
    idxs = np.arange(len(train_samples))
    train_samples = train_samples[idxs]
    train_y = train_y[idxs]


train_samples = np.rollaxis(train_x, 3, 0).astype(np.float32)
test_samples = np.rollaxis(test_x, 3, 0).astype(np.float32)

train_num_examples = len(train_samples)
test_num_examples = len(test_samples)


def datagen(samples=train_samples):

    current_sample = 0

    while (True):
        batch = []
        if (current_sample >= len(samples)-batch_size):
            current_sample = 0
        batch_imgs = samples[current_sample:current_sample+batch_size]
        for i in range(batch_size):
            batch.append(np.copy(batch_imgs[i]).astype(np.float32))
            batch[i] = batch[i].flatten()
        batch_x = np.vstack(batch)
        batch_y = np.stack(
            [i for i in train_y[current_sample:current_sample+batch_size]])
        yield batch_x, batch_y
        current_sample += batch_size


num_batches = (train_num_examples // batch_size)

# <-- Helper Functions --> #


def weight_variable(shape, name=None):
    init = tf.random_normal(shape=shape, stddev=0.2, dtype=tf.float32)
    return tf.Variable(init, dtype=tf.float32, name=name)


def bias_variable(shape, name=None):
    init = tf.constant(value=0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init, dtype=tf.float32, name=name)


def conv2d(x, conv_filter, stride=1, padding='SAME'):
    return tf.nn.conv2d(x, conv_filter, strides=[1, stride, stride, 1], padding='SAME')


def act(x):
    return tf.nn.relu(x)


def sig(x):
    return tf.nn.sigmoid(x)

# <-- Weights --> #


with tf.name_scope('label_in_w'):
    with tf.name_scope('weight_1'):
        label_fc_1 = weight_variable(
            [width*height*channels], name='label_fc_1')
        summary(label_fc_1)
    with tf.name_scope('weight_2'):
        label_fc_2 = weight_variable(
            [width*height*channels, width*height*channels], name='label_fc_2')
        summary(label_fc_2)
    with tf.name_scope('weight_3'):
        label_fc_3 = weight_variable(
            [width*height*channels, width*height*channels], name='label_fc_3')
        summary(label_fc_3)
    with tf.name_scope('weight_conv'):
        label_filter_1 = weight_variable(
            [3, 3, channels, 32], name='label_filter_1')
        summary(label_filter_1)

with tf.name_scope('encoder_weights'):
    with tf.name_scope('conv_1'):
        conv_filter_1 = weight_variable(
            [3, 3, channels, 24], name='gen_filter_1')
        summary(conv_filter_1)
    with tf.name_scope('conv_2'):
        conv_filter_2 = weight_variable([3, 3, 24, 32], name='gen_filter_2')
        summary(conv_filter_2)
    with tf.name_scope('conv_3'):
        conv_filter_3 = weight_variable([3, 3, 32, 32], name='gen_filter_3')
        summary(conv_filter_3)
    with tf.name_scope('conv_4'):
        conv_filter_4 = weight_variable([3, 3, 32, 32], name='gen_filter_4')
        summary(conv_filter_4)
    with tf.name_scope('conv_5'):
        conv_filter_5 = weight_variable([3, 3, 32, 48], name='gen_filter_5')
        summary(conv_filter_5)
    with tf.name_scope('conv_coding'):
        conv_filter_6 = weight_variable(
            [3, 3, 48, target_filters], name='gen_filter_6')
        summary(conv_filter_6)

with tf.name_scope('decoder_weights'):
    with tf.name_scope('conv_1'):
        conv_filter_de1 = weight_variable(
            [3, 3, target_filters, 32], name='gen_de_filter_1')
        summary(conv_filter_de1)
    with tf.name_scope('conv_2'):
        conv_filter_de2 = weight_variable(
            [3, 3, 24, 32], name='gen_de_filter_transpose_2')  # Channel order switched
        summary(conv_filter_de2)
    with tf.name_scope('conv_3'):
        conv_filter_de3 = weight_variable(
            [3, 3, 24, 24], name='gen_de_filter_3')
        summary(conv_filter_de3)
    with tf.name_scope('conv_4'):
        conv_filter_de4 = weight_variable(
            [3, 3, 24, 24], name='gen_de_filter_transpose_4')  # Thanks tensorflow
        summary(conv_filter_de4)
    with tf.name_scope('conv_5'):
        conv_filter_de5 = weight_variable(
            [3, 3, 24, 6], name='gen_de_filter_5')
        summary(conv_filter_de5)
    with tf.name_scope('conv_reconstruct'):
        conv_filter_out = weight_variable(
            [3, 3, 6, channels], name='gen_filter_out')
        summary(conv_filter_out)

with tf.name_scope('encoder_biases'):
    with tf.name_scope('bias_1'):
        bias_1 = bias_variable([24], name='gen_bias_2')
        summary(bias_1)
    with tf.name_scope('bias_2'):
        bias_2 = bias_variable([32], name='gen_bias_2')
        summary(bias_2)
    with tf.name_scope('bias_3'):
        bias_3 = bias_variable([32], name='gen_bias_3')
        summary(bias_3)
    with tf.name_scope('bias_4'):
        bias_4 = bias_variable([32], name='gen_bias_4')
        summary(bias_4)
    with tf.name_scope('bias_5'):
        bias_5 = bias_variable([48], name='gen_bias_5')
        summary(bias_5)
    with tf.name_scope('bias_coding'):
        bias_6 = bias_variable([target_filters], name='gen_bias_6')
        summary(bias_6)

with tf.name_scope('decoder_biases'):
    with tf.name_scope('bias_1'):
        bias_de1 = bias_variable([32], name='gen_de_bias_1')
        summary(bias_de1)
    with tf.name_scope('bias_2'):
        bias_de2 = bias_variable([24], name='gen_de_bias_2')
        summary(bias_de2)
    with tf.name_scope('bias_3'):
        bias_de3 = bias_variable([24], name='gen_de_bias_3')
        summary(bias_de3)
    with tf.name_scope('bias_4'):
        bias_de4 = bias_variable([24], name='gen_de_bias_4')
        summary(bias_de4)
    with tf.name_scope('bias_5'):
        bias_de5 = bias_variable([6], name='gen_de_bias_5')
        summary(bias_de5)
    with tf.name_scope('bias_out'):
        bias_out = bias_variable([channels], name='gen_de_bias_out')
        summary(bias_out)

# <-- Model --> #

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=in_shape)
    y_true = tf.placeholder(tf.float32, shape=out_shape)
    y_true_im = tf.reshape(y_true, shape=[-1, width, height, channels])
    label = tf.placeholder(tf.float32, shape=[None, 1])
    in_reshape = tf.reshape(x, shape=[-1, width, height, channels])

with tf.name_scope('training_boolean'):
    training = tf.placeholder(tf.bool, shape=())

with tf.name_scope('label_in'):
    with tf.name_scope('dense_1'):
        label_fc_in = tf.multiply(label, label_fc_1)
    with tf.name_scope('dense_2'):
        label_fc_in = tf.matmul(label_fc_in, label_fc_2)
    with tf.name_scope('dense_3'):
        label_fc_in = tf.matmul(label_fc_in, label_fc_3)
    with tf.name_scope('image'):
        label_fc_reshape = tf.reshape(
            label_fc_in, shape=[-1, width, height, channels])
    with tf.name_scope('conv_1'):
        label_conv = sig(conv2d(label_fc_reshape, label_filter_1, stride=1))

with tf.name_scope('encoder'):

    with tf.name_scope('conv_1'):
        conv1 = (conv2d(in_reshape, conv_filter_1, stride=1)) + bias_1
    with tf.name_scope('conv_2'):
        conv2 = (conv2d(conv1, conv_filter_2, stride=1) + label_conv)
    with tf.name_scope('conv_3'):
        conv3 = (conv2d(conv2, conv_filter_3, stride=2) + bias_3)
    with tf.name_scope('conv_4'):
        conv4 = (conv2d(conv3, conv_filter_4, stride=2) + bias_4)
    with tf.name_scope('conv_5'):
        conv5 = (conv2d(conv4, conv_filter_5, stride=1) + bias_5)
    with tf.name_scope('conv_6'):
        conv6 = (conv2d(conv5, conv_filter_6, stride=1) + bias_6)

with tf.name_scope('bit-wise_encoding'):
    encoding = tf.round(sig(conv6))

with tf.name_scope('decoder'):

    with tf.name_scope('conv_1'):
        conv_de1 = (conv2d(encoding, conv_filter_de1, stride=1) + bias_de1)

    def training_deconv_1():
        return tf.nn.conv2d_transpose(conv_de1, conv_filter_de2, [batch_size, int(width//2.0), int(height//2.0), 24], strides=[1, 2, 2, 1]) + bias_de2

    def not_training_deconv_1():
        return tf.nn.conv2d_transpose(conv_de1, conv_filter_de2, [1, int(width//2.0), int(height//2.0), 24], strides=[1, 2, 2, 1]) + bias_de2

    def training_deconv_2():
        return tf.nn.conv2d_transpose(conv_de3, conv_filter_de4, [batch_size, width, height, 24], strides=[1, 2, 2, 1]) + bias_de4

    def not_training_deconv_2():
        return tf.nn.conv2d_transpose(conv_de3, conv_filter_de4, [1, width, height, 24], strides=[1, 2, 2, 1]) + bias_de4

    with tf.name_scope('conv_2'):
        conv_de2 = tf.cond(training, training_deconv_1, not_training_deconv_1)
    with tf.name_scope('conv_3'):
        conv_de3 = (conv2d(conv_de2, conv_filter_de3, stride=1) + bias_de3)
    with tf.name_scope('conv_4'):
        conv_de4 = tf.cond(training, training_deconv_2, not_training_deconv_2)
    with tf.name_scope('conv_5'):
        conv_de5 = (conv2d(conv_de4, conv_filter_de5, stride=1) + bias_de5)

with tf.name_scope('reconstruction'):
    out_image = sig(conv2d(conv_de5, conv_filter_out, stride=1) + bias_out)
    out_reshape = tf.reshape(out_image, shape=[-1, width*height*channels])

with tf.name_scope('encoding_alt'):
    coding = tf.placeholder(tf.float32, shape=[None, int(
        width//4.0), int(height//4.0), target_filters])

with tf.name_scope('decoder_alt'):
    # <-- Separate Decoder --> #

    with tf.name_scope('conv_1'):
        conv_decoder_1 = (conv2d(coding, conv_filter_de1, stride=1) + bias_de1)

    def sep_training_deconv_1():
        return tf.nn.conv2d_transpose(conv_decoder_1, conv_filter_de2, [batch_size, int(width//2.0), int(height//2.0), 24], strides=[1, 2, 2, 1]) + bias_de2

    def sep_not_training_deconv_1():
        return tf.nn.conv2d_transpose(conv_decoder_1, conv_filter_de2, [1, int(width//2.0), int(height//2.0), 24], strides=[1, 2, 2, 1]) + bias_de2

    def sep_training_deconv_2():
        return tf.nn.conv2d_transpose(conv_decoder_3, conv_filter_de4, [batch_size, width, height, 24], strides=[1, 2, 2, 1]) + bias_de4

    def sep_not_training_deconv_2():
        return tf.nn.conv2d_transpose(conv_decoder_3, conv_filter_de4, [1, width, height, 24], strides=[1, 2, 2, 1]) + bias_de4

    with tf.name_scope('conv_2'):
        conv_decoder_2 = tf.cond(
            training, sep_training_deconv_1, sep_not_training_deconv_1)
    with tf.name_scope('conv_3'):
        conv_decoder_3 = (
            conv2d(conv_decoder_2, conv_filter_de3, stride=1) + bias_de3)
    with tf.name_scope('conv_4'):
        conv_decoder_4 = tf.cond(
            training, sep_training_deconv_2, sep_not_training_deconv_2)
    with tf.name_scope('conv_5'):
        conv_decoder_5 = (
            conv2d(conv_decoder_4, conv_filter_de5, stride=1) + bias_de5)

with tf.name_scope('reconstruct_alt'):
    out_decoder = sig(
        conv2d(conv_decoder_5, conv_filter_out, stride=1) + bias_out)

# <-- MS-SSIM ( Multi-Scale Structural Similarity Index Metric ) --> #
# - Disclaimer: Shamelessly stolen from stack overflow - #


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    with tf.name_scope('special_gauss'):
        x_data, y_data = np.mgrid[-size//2 +
                                  1:size//2 + 1, -size//2 + 1:size//2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=5, sigma=1.5):
    with tf.name_scope('ssim'):
        window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
        K1 = 0.01
        K2 = 0.03
        L = 1  # depth of image (255 in case the image has a differnt scale)
        C1 = (K1*L)**2
        C2 = (K2*L)**2
        mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
        mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        sigma1_sq = tf.nn.conv2d(
            img1*img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
        sigma2_sq = tf.nn.conv2d(
            img2*img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
        sigma12 = tf.nn.conv2d(
            img1*img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
        if cs_map:
            value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                                                          (sigma1_sq + sigma2_sq + C2)),
                     (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                                                         (sigma1_sq + sigma2_sq + C2))

        if mean_metric:
            value = tf.reduce_mean(value)
        return value

# <-- Automatic Backpropagation Thanks To Tensorflow --> #


trainable_variables = tf.trainable_variables()

# Peak Signal-Noise Ratio (PSNR)
with tf.name_scope('PSNR_loss'):
    loss = tf.multiply(10.0, tf.truediv(
        tf.log(tf.losses.mean_squared_error(out_reshape, y_true)), tf.log(10.0)))

with tf.name_scope('backprop'):
    update_iteration = tf.train.AdamOptimizer(1e-3).minimize(loss)

# <-- Setting Up For Training --> #

sess = tf.InteractiveSession(config=configurations)

# <-- Merging Variable Summaries --> #

merged_summaries = tf.summary.merge_all()
train_write = tf.summary.FileWriter('./logging/log', sess.graph)

sess.run(tf.global_variables_initializer())

# <-- Empty Output Folder --> #

for file in os.scandir("./output/"):
    shutil.rmtree(file)

# <-- File Size Calculation --> #

sample = test_samples[0].astype(np.float32)
encoder_out = encoding.eval(feed_dict={x: np.reshape(
    sample, [1, width*height*channels]), label: np.asarray([[1.0]]), training: False})
print('\nEncoding size (bits): {}\t (bytes): {}\t(target bits): {}'.format(
    encoder_out.size, encoder_out.size / 8., target_bits))

os.mkdir('./output/JPEG')
for idx, test_sample in enumerate(test_samples[0:500]):
    location = './output/JPEG/{}.jpeg'.format(idx)
    sample = Image.fromarray((test_sample*255.0).astype(np.uint8))
    sample.save(location, quality=5)
    sample = np.divide(np.copy(np.asarray(sample)), 255.0)
    jpeg = np.divide(
        np.copy(np.asarray(Image.open(location).convert('RGB'))), 255.0)
    psnr_jpeg = PSNR(jpeg, sample)
    ssim_jpeg = compare_ssim(img_as_float(
        jpeg), img_as_float(sample), multichannel=True)
    print('JPEG Size: {}\t PSNR: {}\t Name: {}'.format(
        os.path.getsize(location), psnr_jpeg, location))
    with open('./jpeg_test_log_data.txt', 'a') as file:
        file.write(location + '\t' + str(ssim_jpeg) + '\n')
print()

# <-- Training Loop --> #

generator = datagen()

for epoch in range(epochs):

    shuffle_train()

    loss_total = 0.

    for batch_idx in range(num_batches):
        batch_x, batch_y = next(generator)

        iter_loss, _ = sess.run([loss, update_iteration], feed_dict={
                                x: batch_x, y_true: batch_x, label: batch_y, training: True})

        loss_total += iter_loss

    loss_total /= float(num_batches)

    print("Epoch: {}\tLoss: {}".format(epoch, loss_total))

    summaries = sess.run(merged_summaries)
    train_write.add_summary(summaries, epoch)

    # Visual Indications
    os.mkdir('./output/{}'.format(str(epoch)))
    test_loss = 0.0
    os.remove('./test_log_data.txt')
    for idx, test_sample in enumerate(test_samples[0:500]):

        sample = np.copy(test_sample).astype(np.float32)
        sample_reshape = np.reshape(sample, [1, width*height*channels])

        encoder_out = encoding.eval(
            feed_dict={x: sample_reshape, label: [test_y[idx]], training: False})
        #encoding_list = [int(i) for i in encoder_out.flatten().tolist()]
        #encoding_bytes = [0 for i in range(floor(target_bits / 8))]
        # for i in range( target_bits // 8 ):
        #	for j in range(8):
        #		encoding_bytes[i] += encoding_list[i*8 + j] * (2**j)
        #encoding_bytes = bytearray(encoding_bytes)
        # with open('./output/{}/{}_coding.bin'.format( str(epoch), str(idx) ), 'wb' ) as file:
        #	file.write(encoding_bytes)
        img = out_decoder.eval(
            feed_dict={coding: encoder_out, training: False})

        test_psnr_loss = PSNR(sample, img)
        test_ssim_loss = compare_ssim(img_as_float(
            sample), img_as_float(img[0]), multichannel=True)
        test_loss += test_psnr_loss

        with open('./test_log_data.txt', 'a') as file:
            file.write('\n./output/{}/{}_Output_Raw.png\t'.format(str(epoch),
                                                                  str(idx)) + str(test_ssim_loss))

        img = (img*255.)[0]
        img = Image.fromarray(img.astype(np.uint8))
        img.save('./output/{}/{}_Output_Raw.png'.format(str(epoch), str(idx)))

        target = sample.reshape([width, height, channels])

        target = Image.fromarray((target*255.0).astype(np.uint8))

        #Image.fromarray( img.astype(np.uint8) ).save('./output/YCC_{}_Output.png'.format( str(epoch) ) )
        #Image.fromarray( target.astype(np.uint8) ).save('./output/YCC_{}_Target.png'.format( str(epoch) ) )

        target.save('./output/{}/{}_Target.png'.format(str(epoch), str(idx)))

    test_loss /= float(len(test_samples[0:500]))
    print("\nTest Loss: {}\n".format(test_loss))
