import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


class Block(layers.Layer):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.h1 = models.Sequential([
            layers.Conv2D(self.filters, kernel_size=self.kernel_size, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.h2 = models.Sequential([
            layers.Conv2D(self.filters, kernel_size=self.kernel_size, strides=1, padding='same'),
            layers.BatchNormalization(),
        ])

    def call(self, x, training=False):
        x_0 = x
        h1 = self.h1(x, training=training)
        h2 = self.h2(h1, training=training)
        out = h2 + x_0
        return out

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.blocks = [Block(self.filters, self.kernel_size) for _ in range(3)]

    def call(self, x, training=False):
        x_0 = x
        for i in range(3):
            x = self.blocks[i](x, training=training)
        out = x + x_0
        return out
    



class Encoder(models.Model):
    def __init__(self, C, filters):
        super().__init__()
        self.C = C
        self.filters = filters
        self.h1 = models.Sequential([
            layers.Conv2D(self.filters // 8, kernel_size=7, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.h2 = models.Sequential([
            layers.Conv2D(self.filters // 6, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.h3 = models.Sequential([
            layers.Conv2D(self.filters // 4, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.h4 = models.Sequential([
            layers.Conv2D(self.filters // 2, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.h5 = models.Sequential([
            layers.Conv2D(self.filters, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.h6 = models.Sequential([
            layers.Conv2D(self.C, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization()
        ])

    def call(self, inputs, training=False):
        h1 = self.h1(inputs, training=training)
        h2 = self.h2(h1, training=training)
        h3 = self.h3(h2, training=training)
        h4 = self.h4(h3, training=training)
        h5 = self.h5(h4, training=training)
        
        return self.h6(h5, training=training)
    



class Generator(models.Model):
    def __init__(self, filters, n_blocks):
        super().__init__()
        self.filters = filters
        self.n_blocks = n_blocks
        self.res_blocks = [ResidualBlock(self.filters, 3) for _ in range(self.n_blocks)]
        self.h1 = models.Sequential([
            layers.Conv2DTranspose(self.filters, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.h2 = models.Sequential([
            layers.Conv2DTranspose(self.filters // 2, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.h3 = models.Sequential([
            layers.Conv2DTranspose(self.filters // 4, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.h4 = models.Sequential([
            layers.Conv2DTranspose(self.filters // 6, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.h5 = models.Sequential([
            layers.Conv2DTranspose(self.filters // 8, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.h6 = models.Sequential([
            layers.Conv2DTranspose(3, kernel_size=7, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Sigmoid()
        ])

    def call(self, inputs, training=False):
        w_hat = self.h1(inputs, training=training)
        for i in range(self.n_blocks):
            w_hat = self.res_blocks[i](w_hat, training=training)
        x_hat = self.h2(x_hat, training=training)
        x_hat = self.h3(x_hat, training=training)
        x_hat = self.h4(x_hat, training=training)
        x_hat = self.h5(x_hat, training=training)
        x_hat = self.h6(x_hat, training=training)
        return x_hat