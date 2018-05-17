__author__ = 'Charlie'
# Utils used with tensorflow implemetation
import tensorflow as tf
import numpy as np
import os, sys
from six.moves import urllib
import tarfile
import zipfile
from skimage import io, color
import scipy.io


def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)


def get_model_data(dir_path, model_url):
    maybe_download_and_extract(dir_path, model_url)
    filename = model_url.split("/")[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    data = scipy.io.loadmat(filepath)
    return data


def save_image(image, save_dir, name):
    """
    Save image by unprocessing and converting to rgb.
    :param image: iamge to save
    :param save_dir: location to save image at
    :param name: prefix to save filename
    :return:
    """
    image = color.lab2rgb(image)
    io.imsave(os.path.join(save_dir, name + ".png"), image)


def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var


def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def get_tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_strided(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv2d_transpose_strided(x, W, b, output_shape=None, stride=2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def leaky_relu(x, alpha=0.2, name=""):
    return tf.maximum(alpha * x, x, name)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5, stddev=0.02):
    """
    Code taken from http://stackoverflow.com/a/34634291/2267819
    """
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                               , trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev),
                                trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed


def process_image(image, mean_pixel):
    return image - mean_pixel


def unprocess_image(image, mean_pixel):
    return image + mean_pixel


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    tf.summary.histogram(var.op.name + "/activation", var)
    tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)


"""
The residual code below is taken and modified
 from https://github.com/tensorflow/models/blob/master/resnet/resnet_model.py
"""


def residual_block(x, in_filter, out_filter, stride, phase_train, is_conv=True, leakiness=0.0,
                   activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
        with tf.variable_scope('shared_activation'):
            x = batch_norm(x, out_filter, phase_train, scope="init_bn")
            x = leaky_relu(x, alpha=leakiness, name="lrelu")
            orig_x = x
    else:
        with tf.variable_scope('residual_only_activation'):
            orig_x = x
            x = batch_norm(x, out_filter, phase_train, scope="init_bn")
            x = leaky_relu(x, alpha=leakiness, name="lrelu")

    with tf.variable_scope('sub1'):
        if is_conv:
            x = conv_no_bias('conv1', x, 3, in_filter, out_filter, stride)
        else:
            x = conv_transpose_no_bias('conv_t1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
        x = batch_norm(x, out_filter, phase_train, scope="bn2")
        x = tf.nn.relu(x, "relu")
        if is_conv:
            x = conv_no_bias('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
        else:
            x = conv_transpose_no_bias('conv_t2', x, 3, in_filter, in_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
        if in_filter != out_filter:
            if is_conv:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
            else:
                orig_x = tf.nn.fractional_avg_pool(orig_x, stride)  # Available only in tf 0.11 - not tested
            orig_x = tf.pad(
                orig_x, [[0, 0], [0, 0], [0, 0],
                         [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
        x += orig_x

    # tf.logging.info('image after unit %s', x.get_shape())
    return x


def bottleneck_residual_block(x, in_filter, out_filter, stride, phase_train, is_conv=True, leakiness=0.0,
                              activate_before_residual=False):
    """Bottleneck resisual unit with 3 sub layers."""
    if activate_before_residual:
        with tf.variable_scope('common_bn_relu'):
            x = batch_norm(x, out_filter, phase_train, scope="init_bn")
            x = leaky_relu(x, alpha=leakiness, name="lrelu")
            orig_x = x
    else:
        with tf.variable_scope('residual_bn_relu'):
            orig_x = x
            x = batch_norm(x, out_filter, phase_train, scope="init_bn")
            x = leaky_relu(x, alpha=leakiness, name="lrelu")

    with tf.variable_scope('sub1'):
        if is_conv:
            x = conv_no_bias('conv1', x, 1, in_filter, out_filter / 4, stride)
        else:
            x = conv_transpose_no_bias('conv_t1', x, 1, out_filter / 4, out_filter, stride)

    with tf.variable_scope('sub2'):
        x = batch_norm(x, out_filter, phase_train, scope="bn2")
        x = leaky_relu(x, alpha=leakiness, name="lrelu")
        if is_conv:
            x = conv_no_bias('conv2', x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])
        else:
            x = conv_transpose_no_bias('conv_t2', x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
        x = batch_norm(x, out_filter, phase_train, scope="bn3")
        x = leaky_relu(x, alpha=leakiness, name="lrelu")
        if is_conv:
            x = conv_no_bias('conv3', x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])
        else:
            x = conv_transpose_no_bias('conv_t3', x, 1, in_filter, out_filter / 4, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
        if in_filter != out_filter:
            if is_conv:
                orig_x = conv_no_bias('project', orig_x, 1, in_filter, out_filter, stride)
            else:
                orig_x = conv_transpose_no_bias('project', orig_x, 1, in_filter, out_filter, stride)
        x += orig_x

    # tf.logging.info('image after unit %s', x.get_shape())
    return x


def conv_no_bias(name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable(
            'DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0 / n)))
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')


def conv_transpose_no_bias(name, x, filter_size, in_filters, out_filters, strides):
    """Convolution Transpose."""
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable(
            'DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0 / n)))

        output_shape = tf.shape(x)
        output_shape[1] *= strides[1]
        output_shape[2] *= strides[2]
        output_shape[3] = in_filters

        return tf.nn.conv2d_transpose(x, kernel, output_shape=output_shape, strides=strides, padding='SAME')
