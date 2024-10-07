from PIL import Image

import tensorflow as tf

def webp2bmp(img):
    pass

def normalize_input(x):
    # valores baseados no conjunto de treino do cityscapes
    _mean = tf.convert_to_tensor([72., 82., 72.])
    _var = tf.convert_to_tensor([2060., 2160., 2110.])
    return (x - _mean) / tf.math.sqrt(_var + 1e-10)

def denormalize_output(x_hat):
    # valores baseados no conjunto de treino do cityscapes
    _mean = tf.convert_to_tensor([72., 82., 72.])
    _var = tf.convert_to_tensor([2060., 2160., 2110.])
    x = x_hat * tf.math.sqrt(_var + 1e-10) + _mean
    x = tf.clip_by_value(x, clip_value_min=0, clip_value_max=255)
    return x