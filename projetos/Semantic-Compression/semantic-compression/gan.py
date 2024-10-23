import numpy as np
import tensorflow as tf
from keras import layers, models
# from utils import normalize_input, denormalize_output


class Block(layers.Layer):
    def __init__(self, filters, kernel_size, momentum=0.99):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.h1 = models.Sequential([
            layers.Conv2D(self.filters, kernel_size=self.kernel_size, strides=1, padding='same'),
            layers.BatchNormalization(momentum=momentum),
            layers.ReLU()
        ])
        self.h2 = models.Sequential([
            layers.Conv2D(self.filters, kernel_size=self.kernel_size, strides=1, padding='same'),
            layers.BatchNormalization(momentum=momentum),
        ])

    def call(self, x, training=False):
        x_0 = x
        h1 = self.h1(x, training=training)
        h2 = self.h2(h1, training=training)
        out = h2 + x_0
        return out


class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, momentum=0.99):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.blocks = [Block(self.filters, self.kernel_size, momentum=momentum) for _ in range(3)]

    def call(self, x, training=False):
        x_0 = x
        for i in range(3):
            x = self.blocks[i](x, training=training)
        out = x + x_0
        return out


class Encoder(models.Model):
    """
    input: k7s1 [filters / 2**n_convs]
    downsample: k5s2 [filters / 2**(n_convs-i)], i=1,2,...,n_convs
    output: k3s1 [channels]
    """
    def __init__(self, channels, filters, n_convs=4, momentum=0.99):
        super().__init__()
        self.channels = channels
        self.filters = filters
        self.n_convs = n_convs
        self.h_in = models.Sequential([
            layers.Conv2D(
                self.filters // 2**self.n_convs,
                kernel_size=7, strides=1, padding='same'
            ),
            layers.BatchNormalization(monentum=momentum),
            layers.ReLU()
        ])
        self.downsample = models.Sequential()
        for i in range(self.n_convs):
            self.downsample.add(
                layers.Conv2D(
                    self.filters // 2**(self.n_convs - i - 1),
                    kernel_size=5, strides=2, padding='same'
                )
            )
            self.downsample.add(layers.BatchNormalization(momentum=momentum))
            self.downsample.add(layers.ReLU())
        self.h_out = models.Sequential([
            layers.Conv2D(self.channels, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(momentum=momentum)
        ])

    def call(self, inputs, training=False):
        h_in = self.h_in(inputs, training=training)
        h_d = self.downsample(h_in, training=training)
        w = self.h_out(h_d, training=training)
        return w


class Quantizer(layers.Layer):
    def __init__(self, sigma, levels, l_min, l_max):
        super().__init__()
        self.sigma = sigma
        self.levels = levels
        self.l_min = l_min
        self.l_max = l_max

    def call(self, z):
        # convert z -> (B, C, H, W)
        z_t = tf.transpose(z, (0, 3, 1, 2))
        z_t_shape = tf.shape(z_t)
        B = z_t_shape[0]
        C = z_t_shape[1]
        z_vec = tf.reshape(z_t, [B, C, -1, 1])
        # definir array centers
        centers = tf.linspace(float(-2), float(2), 5)
        # definir tensor dist
        dist = tf.square(tf.abs(z_vec - centers))
        # calcular tensor z_hat
        phi = tf.nn.softmax(-self.sigma * dist, axis=-1)
        z_hat = tf.reshape(tf.reduce_sum(phi * centers, axis=3), z_t_shape)
        z_hat = tf.transpose(z_hat, (0, 2, 3, 1))
        return z_hat


class Generator(models.Model):
    """
    input: k3s1 [filters]
    resnet: n_blocks x (k3s1 [filters] + k3s1 [filters])
    upsample: k5s2 [filters / 2**i], i=1,2,...,n_convs
    output: k7s1 [3]
    """
    def __init__(self, n_blocks, filters, n_convs=4, momentum=0.99):
        super().__init__()
        self.filters = filters
        self.n_blocks = n_blocks
        self.res_blocks = [ResidualBlock(self.filters, kernel_size=3, momentum=momentum) for _ in range(self.n_blocks)]
        self.h_in = models.Sequential([
            layers.Conv2DTranspose(self.filters, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(momentum=momentum),
            layers.ReLU()
        ])
        self.upsample = models.Sequential()
        for i in range(n_convs):
            self.upsample.add(
                layers.Conv2DTranspose(
                    self.filters // 2**(i + 1),
                    kernel_size=5, strides=2, padding='same'
                )
            )
            self.upsample.add(layers.BatchNormalization(momentum=momentum))
            self.upsample.add(layers.ReLU())
        self.h_out = models.Sequential([
            layers.Conv2DTranspose(3, kernel_size=7, strides=1, padding='same'),
            layers.BatchNormalization(momentum=momentum),
        ])

    def call(self, inputs, training=False):
        h = self.h_in(inputs, training=training)
        for i in range(self.n_blocks):
            h = self.res_blocks[i](h, training=training)
        h_up = self.upsample(h, training=training)
        x_hat = self.h_out(h_up, training=training)
        return x_hat


class AutoEncoder(models.Model):
    def __init__(self, filters, n_blocks, channels, n_convs, sigma, levels, l_min, l_max, momentum=0.99):
        super().__init__()
        self.encoder = Encoder(channels, filters, n_convs, momentum=momentum)
        self.decoder = Generator(n_blocks, filters, n_convs, momentum=momentum)
        self.quantizer = Quantizer(sigma, levels, l_min, l_max)

    def call(self, x, training=False):
        w = self.encoder(x, training=training)
        w_hat = self.quantizer(w)
        x_hat = self.decoder(w_hat, training=training)
        return x_hat


class Discriminator(models.Model):
    """
    downsample: k4s2 [filters / 2**(n_convs-i)], i=1,2,...,n_convs
    output: k4s1 [1] + dense(sigmoid)
    """
    def __init__(self, filters, n_convs=4, momentum=0.99):
        super().__init__()
        self.filters = filters
        self.n_convs = n_convs
        self.net = models.Sequential()
        for i in range(self.n_convs):
            self.net.add(
                layers.Conv2D(
                    self.filters // 2**(self.n_convs - i - 1),
                    kernel_size=4, strides=2, padding='same'
                )
            )
            self.net.add(layers.BatchNormalization(momentum=momentum))
            self.net.add(layers.LeakyReLU(0.2))
        self.h_out = models.Sequential([
            layers.Conv2D(1, kernel_size=4, strides=1, padding='same'),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, training=False):
        features = self.net(inputs, training=training)
        y = self.h_out(features, training=training)
        return y


class NegativeGradientLayer(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        @tf.custom_gradient
        def identity_with_neg_grad(x):
            def grad(dy):
                # Return the negative of the incoming gradient
                return -dy
            return x, grad
        # Apply the custom gradient operation to the inputs
        return identity_with_neg_grad(inputs)


class GCGAN(models.Model):
    def __init__(self, filters, n_blocks, channels, n_convs, *, sigma=1000, levels=5, l_min=-2, l_max=2, lambda_d=10, momentum=0.99, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.n_blocks = n_blocks
        self.channels = channels
        self.n_convs = n_convs
        self.sigma = sigma
        self.levels = levels
        self.l_min = l_min
        self.l_max = l_max
        self.lambda_d = lambda_d
        self.momentum = momentum
        self.preprocess = layers.Rescaling(1./255.)
        self.autoencoder = AutoEncoder(filters, n_blocks, channels, n_convs, sigma, levels, l_min, l_max, momentum)
        self.reverse_layer = NegativeGradientLayer()
        self.discriminator = Discriminator(filters, n_convs)
        self.loss_fn = least_squares_gan_loss

    def compile(self, optimizer_1, optimizer_2, **kwargs):
        super().compile(**kwargs)
        self.opt1 = optimizer_1
        self.opt2 = optimizer_2

    def call(self, x, training=False):
        x0 = self.preprocess(x)
        x_hat = self.autoencoder(x0, training=training)
        x_tilde = self.reverse_layer(x_hat)
        y = self.discriminator(x, training=training)
        y_hat = self.discriminator(x_tilde)
        return x0, x_hat, y, y_hat

    def train_step(self, x, discriminating=True):
        with tf.GradientTape() as tape:
            x0, x_hat, y, y_hat = self(x, training=True)
            gan_loss, l2_loss = self.loss_fn(x0, x_hat, y, y_hat)
            if discriminating:
                loss = self.lambda_d * l2_loss - gan_loss
                train_vars = self.trainable_variables
                opt = self.opt2
            else:
                loss = l2_loss
                train_vars = self.autoencoder.trainable_variables
                opt = self.opt1
            grads = tape.gradient(loss, train_vars)
            opt.apply_gradients(zip(grads, train_vars))
        return gan_loss, l2_loss

    def test_step(self, x):
        x0, x_hat, y, y_hat = self(x)
        return self.loss_fn(x0, x_hat, y, y_hat)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "n_blocks": self.n_blocks,
                "channels": self.channels,
                "n_convs": self.n_convs,
                "sigma": self.sigma,
                "levels": self.levels,
                "l_min": self.l_min,
                "l_max": self.l_max,
                "lambda_d": self.lambda_d,
                "momentum": self.momentum
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            config['filters'], config['n_blocks'], config['channels'], config['n_convs'],
            sigma=config['sigma'], levels=config['levels'],
            l_min=config['l_min'], l_max=config['l_max'],
            lambda_d=config['lambda_d'], momentum=config['momentum']
        )


def least_squares_gan_loss(x, x_hat, y, y_hat):
    gan_loss = tf.reduce_mean(y**2) + tf.reduce_mean((y_hat - 1)**2)
    l2_loss =  tf.reduce_mean((x - x_hat)**2)
    return gan_loss, l2_loss

