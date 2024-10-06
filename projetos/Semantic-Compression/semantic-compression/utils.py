# store useful variables and configuration

from PIL import Image

import tensorflow as tf

def webp2bmp(img):
    pass

def normalize_input(x):
    _mean = tf.convert_to_tensor([115., 114., 112.])
    _var = tf.convert_to_tensor([1450., 1510., 1700.])
    return (x - _mean) / tf.math.sqrt(_var + 1e-10)

def denormalize_output(x0_hat):
    _mean = tf.convert_to_tensor([115., 114., 112.])
    _var = tf.convert_to_tensor([1450., 1510., 1700.])
    x_hat = x0_hat * tf.math.sqrt(_var + 1e-10) + _mean
    x_hat = tf.clip_by_value(x_hat, clip_value_min=0, clip_value_max=255)
    return x_hat