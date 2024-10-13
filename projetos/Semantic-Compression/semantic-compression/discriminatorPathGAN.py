import tensorflow as tf
from tensorflow.keras import layers, models

class PatchGANDiscriminator(models.Model):
    def __init__(self, filters=64):
        super().__init__()
        self.filters = filters
        self.net = models.Sequential([
            layers.Conv2D(self.filters, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(0.2), 
            layers.Conv2D(self.filters * 2, kernel_size=4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(self.filters * 4, kernel_size=4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(self.filters * 8, kernel_size=4, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1, kernel_size=4, strides=1, padding='same')
        ])

    def call(self, inputs, training=False):
        output = self.net(inputs, training=training)
        output_mean = tf.reduce_mean(output)
        return output_mean