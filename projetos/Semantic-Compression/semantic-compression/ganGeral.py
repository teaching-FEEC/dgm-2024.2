import numpy as np
import tensorflow as tf
from keras import layers, models
import tensorflow_addons as tfa
# from utils import normalize_input, denormalize_output


class Block(layers.Layer):
    def __init__(self, filters, kernel_size, momentum=0.99):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.h1 = models.Sequential([
            layers.Conv2D(self.filters, kernel_size=self.kernel_size, strides=1, padding='same'),
            tfa.layers.InstanceNormalization(),
            layers.ReLU()
        ])
        self.h2 = models.Sequential([
            layers.Conv2D(self.filters, kernel_size=self.kernel_size, strides=1, padding='same'),
            tfa.layers.InstanceNormalization(),
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
            tfa.layers.InstanceNormalization(),
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
            self.downsample.add(tfa.layers.InstanceNormalization())
            self.downsample.add(layers.ReLU())
        self.h_out = models.Sequential([
            layers.Conv2D(self.channels, kernel_size=3, strides=1, padding='same'),
            tfa.layers.InstanceNormalization()
        ])

    def call(self, inputs, training=False):
        h_in = self.h_in(inputs, training=training)
        h_d = self.downsample(h_in, training=training)
        w = self.h_out(h_d, training=training)
        return w
    
class SemanticEncoder(models.Model):
    """
    input: k7s1 [filters / 2**n_convs]
    downsample: k5s2 [filters / 2**(n_convs-i)], i=1,2,...,n_convs
    """
    def __init__(self, filters, n_convs=4, momentum=0.99):
        super().__init__()
        # self.channels = channels
        self.filters = filters
        self.n_convs = n_convs
        self.h_in = models.Sequential([
            layers.Conv2D(
                self.filters // 2**self.n_convs,
                kernel_size=7, strides=1, padding='same'
            ),
            tfa.layers.InstanceNormalization(),
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
            self.downsample.add(tfa.layers.InstanceNormalization())
            self.downsample.add(layers.ReLU())
        # self.h_out = models.Sequential([
        #     layers.Conv2D(self.channels, kernel_size=3, strides=1, padding='same'),
        #     tfa.layers.InstanceNormalization()
        # ])

    def call(self, inputs, training=False):
        h_in = self.h_in(inputs, training=training)
        h_d = self.downsample(h_in, training=training)
        # w = self.h_out(h_d, training=training)
        return h_d


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
            tfa.layers.InstanceNormalization(),
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
            self.upsample.add(tfa.layers.InstanceNormalization())
            self.upsample.add(layers.ReLU())
        self.h_out = models.Sequential([
            layers.Conv2DTranspose(3, kernel_size=7, strides=1, padding='same'),
            #tfa.layers.InstanceNormalization(),
            layers.Activation('sigmoid')
        ])

    def call(self, inputs, training=False):
        h = self.h_in(inputs, training=training)
        for i in range(self.n_blocks):
            h = self.res_blocks[i](h, training=training)
        h_up = self.upsample(h, training=training)
        x_hat = self.h_out(h_up, training=training)
        return x_hat


class AutoEncoder(models.Model):
    def __init__(self, filters, n_blocks, channels, n_convs, sigma, levels, l_min, l_max, momentum=0.99, SC = 0):
        super().__init__()
        self.filters = filters
        self.n_blocks = n_blocks
        self.channels = channels
        self.n_convs = n_convs
        self.sigma = sigma
        self.levels = levels
        self.l_min = l_min
        self.l_max = l_max
        self.momentum = momentum
        self.SC = SC
        self.encoder = Encoder(channels, filters, n_convs, momentum=momentum)
        self.semanticEncoder = SemanticEncoder(filters, n_convs)
        self.decoder = Generator(n_blocks, filters, n_convs, momentum=momentum)
        self.quantizer = Quantizer(sigma, levels, l_min, l_max)

    def call(self, x, xs, training=False):
        if self.SC == 0:
            w = self.encoder(x, training=training)
            w_hat = self.quantizer(w)
            x_hat = self.decoder(w_hat, training=training)

        elif self.SC == 1:
            x_concat = tf.concat([x, xs], axis=-1)
            w = self.encoder(x_concat, training=training)
            w_hat = self.quantizer(w)
            x_hat = self.decoder(w_hat, training=training)
        
        else:
            x_concat = tf.concat([x, xs], axis=-1)
            w = self.encoder(x_concat, training=training)
            w_hat = self.quantizer(w)
            xs_hat = self.semanticEncoder(xs, training=training)
            w_hat = tf.concat([w_hat, xs_hat], axis=-1)
            x_hat = self.decoder(w_hat, training=training)
        return x_hat

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
            momentum=config['momentum']
        )


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
            if i >= 1:
                self.net.add(tfa.layers.InstanceNormalization())
            self.net.add(layers.LeakyReLU(0.2))
            self.net.add(layers.Dropout(0.3))
        self.h_out = models.Sequential([
            layers.Conv2D(1, kernel_size=4, strides=1, padding='same'),
            layers.Activation('sigmoid')
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
    def __init__(self, filters, n_blocks, channels, n_convs, *, sigma=1000, levels=5, l_min=-2, l_max=2, lambda_d=10, momentum=0.99, SC = 0, **kwargs):
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
        self.SC = SC
        self.preprocess = layers.Rescaling(1./255.)
        self.autoencoder = AutoEncoder(filters, n_blocks, channels, n_convs, sigma, levels, l_min, l_max, momentum, SC)
        #self.semanticEncoder = SemanticEncoder(filters, n_convs)
        # self.reverse_layer = NegativeGradientLayer()
        self.discriminator = Discriminator(filters, n_convs)
        # self.loss_fn = least_squares_gan_loss

    def compile(self, optimizer_1, optimizer_2, **kwargs):
        super().compile(**kwargs)
        self.opt1 = optimizer_1
        self.opt2 = optimizer_2

    def call(self, x, training=False):
        x0 = self.preprocess(x)

        if training:
            x_hat = self.autoencoder(x0, training=False)
            # x_tilde = self.reverse_layer(x_hat)
            #x0_noisy = x0 + tf.random.normal(shape=tf.shape(x0), mean=0.0, stddev=0.1)
            #x_hat_noisy = x_hat + tf.random.normal(shape=tf.shape(x_hat), mean=0.0, stddev=0.1)

            y = self.discriminator(x0, training=True)
            y_hat = self.discriminator(x_hat, training=True)
        else:
            x_hat = self.autoencoder(x0, training=True)
            # x_tilde = self.reverse_layer(x_hat)
            #x0_noisy = x0 + tf.random.normal(shape=tf.shape(x0), mean=0.0, stddev=0.1)
            #x_hat_noisy = x_hat + tf.random.normal(shape=tf.shape(x_hat), mean=0.0, stddev=0.1)

            y = self.discriminator(x0, training=False)
            y_hat = self.discriminator(x_hat, training=False)

        return x0, x_hat, y, y_hat

    def train_step(self, x, xs, discriminating=True):

      if discriminating:

          with tf.GradientTape() as tape:
              x0 = self.preprocess(x)
              x0_concat = tf.concat([x0, xs], axis=-1)

              x_hat = self.autoencoder(x0, xs, training=False)
              x_hat_concat = tf.concat([x_hat, xs], axis=-1)

              if  self.SC == 0:
                    x0_noisy = x0 + tf.random.normal(shape=tf.shape(x0), mean=0.0, stddev=0.1)
                    x_hat_noisy = x_hat + tf.random.normal(shape=tf.shape(x_hat), mean=0.0, stddev=0.1)

              else:
                    x0_noisy = x0_concat + tf.random.normal(shape=tf.shape(x0_concat), mean=0.0, stddev=0.1)
                    x_hat_noisy = x_hat_concat + tf.random.normal(shape=tf.shape(x_hat_concat), mean=0.0, stddev=0.1)

              y = self.discriminator(x0_noisy, training=True)
              y_hat = self.discriminator(x_hat_noisy, training=True)

              real_labels = tf.ones_like(y) *0.95
              fake_labels = tf.zeros_like(y_hat) +0.1

              # # Perda do discriminador
              disc_loss_real = 0.5 * tf.reduce_mean(tf.square(y - real_labels))
              disc_loss_fake = 0.5 * tf.reduce_mean(tf.square(y_hat - fake_labels))
              total_loss = disc_loss_real + disc_loss_fake


              # disc_loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=False)(real_labels, y)
              # disc_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits=False)(fake_labels, y_hat)
              # total_loss = (disc_loss_real + disc_loss_fake) / 2


              grads = tape.gradient(total_loss, self.discriminator.trainable_variables)
              self.opt2.apply_gradients(zip(grads, self.discriminator.trainable_variables))

          with tf.GradientTape() as tape:
              x0 = self.preprocess(x)
              x0_concat = tf.concat([x0, xs], axis=-1)

              x_hat = self.autoencoder(x0_concat, training=True)
              x_hat_concat = tf.concat([x_hat, xs], axis=-1)

              if  self.SC == 0:
                    x0_noisy = x0 + tf.random.normal(shape=tf.shape(x0), mean=0.0, stddev=0.1)
                    x_hat_noisy = x_hat + tf.random.normal(shape=tf.shape(x_hat), mean=0.0, stddev=0.1)

              else:
                    x0_noisy = x0_concat + tf.random.normal(shape=tf.shape(x0_concat), mean=0.0, stddev=0.1)
                    x_hat_noisy = x_hat_concat + tf.random.normal(shape=tf.shape(x_hat_concat), mean=0.0, stddev=0.1)

              y_hat = self.discriminator(x_hat_noisy, training=False)
              y = self.discriminator(x0_noisy, training=False)

              real_labels = tf.ones_like(y) *0.95

              # Perda do gerador
              adv_loss = 0.5 * tf.reduce_mean(tf.square(y_hat - real_labels))
              l2_loss = tf.reduce_mean(tf.square(x0- x_hat))
              loss = self.lambda_d * l2_loss + adv_loss

              # Compute generator adversarial loss
              # adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(real_labels, y_hat)
              # l2_loss = tf.reduce_mean(tf.square(x0 - x_hat))
              # loss = self.lambda_d * l2_loss + adv_loss

              grads = tape.gradient(loss, self.autoencoder.trainable_variables)
              self.opt1.apply_gradients(zip(grads, self.autoencoder.trainable_variables))

          return adv_loss, l2_loss, total_loss, loss

      else:

          with tf.GradientTape() as tape:
              x0 = self.preprocess(x)
              x0_concat = tf.concat([x0, xs], axis=-1)

              x_hat = self.autoencoder(x0_concat, training=False)
              x_hat_concat = tf.concat([x_hat, xs], axis=-1)

              if  self.SC == 0:
                    x0_noisy = x0 + tf.random.normal(shape=tf.shape(x0), mean=0.0, stddev=0.1)
                    x_hat_noisy = x_hat + tf.random.normal(shape=tf.shape(x_hat), mean=0.0, stddev=0.1)

              else:
                    x0_noisy = x0_concat + tf.random.normal(shape=tf.shape(x0_concat), mean=0.0, stddev=0.1)
                    x_hat_noisy = x_hat_concat + tf.random.normal(shape=tf.shape(x_hat_concat), mean=0.0, stddev=0.1)

              y = self.discriminator(x0_noisy, training=True)
              y_hat = self.discriminator(x_hat_noisy, training=True)

              real_labels = tf.ones_like(y) *0.95
              fake_labels = tf.zeros_like(y_hat)+0.1

              # # Perda do discriminador
              disc_loss_real = 0.5 * tf.reduce_mean(tf.square(y - real_labels))
              disc_loss_fake = 0.5 * tf.reduce_mean(tf.square(y_hat - fake_labels))
              total_loss = disc_loss_real + disc_loss_fake


              # disc_loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=False)(real_labels, y)
              # disc_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits=False)(fake_labels, y_hat)
              # total_loss = (disc_loss_real + disc_loss_fake) / 2


              grads = tape.gradient(total_loss, self.discriminator.trainable_variables)
              self.opt2.apply_gradients(zip(grads, self.discriminator.trainable_variables))

          with tf.GradientTape() as tape:
              # # Compute reconstruction loss
              x0 = self.preprocess(x)
              x0_concat = tf.concat([x0, xs], axis=-1)

              x_hat = self.autoencoder(x0_concat, training=False)
              x_hat_concat = tf.concat([x_hat, xs], axis=-1)

              if  self.SC == 0:
                    x0_noisy = x0 + tf.random.normal(shape=tf.shape(x0), mean=0.0, stddev=0.1)
                    x_hat_noisy = x_hat + tf.random.normal(shape=tf.shape(x_hat), mean=0.0, stddev=0.1)

              else:
                    x0_noisy = x0_concat + tf.random.normal(shape=tf.shape(x0_concat), mean=0.0, stddev=0.1)
                    x_hat_noisy = x_hat_concat + tf.random.normal(shape=tf.shape(x_hat_concat), mean=0.0, stddev=0.1)

              y_hat = self.discriminator(x_hat_noisy, training=False)
              y = self.discriminator(x0_noisy, training=False)

              real_labels = tf.ones_like(y)*0.95

              # Perda do gerador
              adv_loss = 0.5 * tf.reduce_mean(tf.square(y_hat - real_labels))
              l2_loss = tf.reduce_mean(tf.square(x0_concat - x_hat_concat))
              loss = self.lambda_d * l2_loss + adv_loss

              # Compute generator adversarial loss
              # adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(real_labels, y_hat)
              # l2_loss = tf.reduce_mean(tf.square(x0 - x_hat))
              # loss = self.lambda_d * l2_loss + adv_loss

          return adv_loss, l2_loss, total_loss, loss

    def test_step(self, x, xs):
        x0 = self.preprocess(x)
        x0_concat = tf.concat([x0, xs], axis=-1)

        x_hat = self.autoencoder(x0, training=False)
        x_hat_concat = tf.concat([x_hat, xs], axis=-1)

        if self.SC == 0:
                y = self.discriminator(x0, training=False)
                y_hat = self.discriminator(x_hat, training=False)
        else:
                y = self.discriminator(x0_concat, training=False)
                y_hat = self.discriminator(x_hat_concat, training=False) 

        real_labels = tf.ones_like(y) *0.95
        fake_labels = tf.zeros_like(y_hat) +0.1

        real_loss = tf.reduce_mean((y - real_labels) ** 2)
        fake_loss = tf.reduce_mean(y_hat ** 2)
        total_loss = 0.5 * (real_loss + fake_loss)

        l2_loss = tf.reduce_mean(tf.square(x0_concat - x_hat_concat))
        adv_loss = 0.5 * tf.reduce_mean((y_hat - real_labels) ** 2)
        loss = self.lambda_d * l2_loss + adv_loss


        # disc_loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=False)(real_labels, y)
        # disc_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits=False)(fake_labels, y_hat)
        # total_loss = (disc_loss_real + disc_loss_fake) / 2

        # adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(real_labels, y_hat)

        # l2_loss = tf.reduce_mean(tf.square(x0 - x_hat))
        # loss = self.lambda_d * l2_loss + adv_loss

        return adv_loss, l2_loss, total_loss, loss




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


# def least_squares_gan_loss(x, x_hat, y, y_hat):
#     gan_loss = tf.reduce_mean(y**2) + tf.reduce_mean((y_hat - 1)**2)
#     l2_loss =  tf.reduce_mean((x - x_hat)**2)
#     return gan_loss, l2_loss



