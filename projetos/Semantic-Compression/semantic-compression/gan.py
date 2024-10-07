import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from keras import layers, models, optimizers
from utils import normalize_input, denormalize_output

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
    def __init__(self, C, filters, n_blocks):
        super().__init__()
        self.C = C
        self.filters = filters
        self.n_blocks = n_blocks
        self.res_blocks = [ResidualBlock(self.filters, 3) for _ in range(self.n_blocks)]
        self.h1 = models.Sequential([
            layers.Conv2D(self.filters // 2, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.h2 = models.Sequential([
            layers.Conv2D(self.filters, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.block = Block(self.filters, 3)
        self.h3 = models.Sequential([
            layers.Conv2D(self.C, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization()
        ])

    def call(self, inputs, training=False):
        h1 = self.h1(inputs, training=training)
        x = self.h2(h1, training=training)
        x_0 = x
        for i in range(self.n_blocks):
            x = self.res_blocks[i](x, training=training)
        x = self.block(x, training=training)
        w = x + x_0
        w = self.h3(w, training=training)
        return w


class Generator(models.Model):
    def __init__(self, filters, n_blocks):
        super().__init__()
        self.filters = filters
        self.n_blocks = n_blocks
        self.res_blocks = [ResidualBlock(self.filters, 3) for _ in range(self.n_blocks)]
        self.h1 = models.Sequential([
            layers.Conv2DTranspose(self.filters, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.block = Block(self.filters, 3)
        self.h2 = models.Sequential([
            layers.Conv2DTranspose(self.filters // 2, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.h3 = models.Sequential([
            layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization()
        ])

    def call(self, inputs, training=False):
        w_hat = self.h1(inputs, training=training)
        w_0 = w_hat
        for i in range(self.n_blocks):
            w_hat = self.res_blocks[i](w_hat, training=training)
        w_hat = self.block(w_hat, training=training)
        x_hat = w_hat + w_0
        x_hat = self.h2(x_hat, training=training)
        x_hat = self.h3(x_hat, training=training)
        return x_hat


class Discriminator(models.Model):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters
        self.net = models.Sequential([
            layers.Conv2D(self.filters, kernel_size=4, strides=2, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(self.filters // 2, kernel_size=4, strides=2, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(self.filters // 4, kernel_size=4, strides=2, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, training=False):
        out = self.net(inputs, training=training)
        return out


def quantizer(L, c_min, c_max):
    """
    z: ndarray(B, H, W, C)
    L: int
    """
    def q(z):
        # convert z -> (B, C, H, W)
        z_t = tf.transpose(z, (0, 3, 1, 2))
        z_t_shape = tf.shape(z_t)
        B, C = z_t_shape[:2]
        z_vec = tf.reshape(z_t, [B, C, -1, 1])
        # definir array centers
        centers = np.linspace(float(c_min), float(c_max), L)
        # definir tensor dist
        dist = tf.square(tf.abs(z_vec - centers))
        # calcular tensor z_hat
        symbols = tf.reshape(tf.argmin(dist, axis=-1), z_t_shape)
        symbols = tf.transpose(symbols, (0, 2, 3, 1))
        z_hat = tf.convert_to_tensor(centers[symbols])
        return z_hat, symbols
    return q


class AutoEncoder(models.Model):
    def __init__(self, L, e_filters, e_blocks, g_filters, g_blocks, c_min, c_max):
        super().__init__()
        self.encoder = Encoder(L, e_filters, e_blocks)
        self.decoder = Generator(g_filters, g_blocks)
        self.quantizer = quantizer(L, c_min, c_max)

    def call(self, x, training=False):
        w = self.encoder(x, training=training)
        w_hat, _ = self.quantizer(w)
        x_hat = self.decoder(w_hat, training=training)
        return x_hat, w_hat


class GenerativeCompressionGAN(models.Model):
    def __init__(self, e_filters, e_blocks, g_filters, g_blocks, d_filters, L=5, c_min=-2, c_max=2, lambda_d=10):
        super().__init__()
        self.autoencoder = AutoEncoder(L, e_filters, e_blocks, g_filters, g_blocks, c_min, c_max)
        self.discriminator = Discriminator(d_filters)
        self.quantizer = quantizer(L=L, c_min=c_min, c_max=c_max)
        self.lambda_d = lambda_d  # Peso para a perda de distorção
        self.L = L  # Número de níveis de quantização

    def compile(self, d_optimizer, g_optimizer, gan_loss_fn, distortion_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.gan_loss_fn = gan_loss_fn  # Perda adversarial
        self.distortion_loss_fn = distortion_loss_fn  # Perda de distorção

    def call(self, data):
        real_images = normalize_input(data)
        generated_images, quantized_latent = self.autoencoder(real_images)
        return denormalize_output(generated_images), quantized_latent

    def evaluate(self, dataloader_val):
        d_loss_val = 0.
        g_loss_val = 0.
        for data in tqdm(dataloader_val):
            real_images = normalize_input(data)
            generated_images, quantized_latent = self.autoencoder(real_images)
            real_output = self.discriminator(real_images)
            fake_output = self.discriminator(generated_images)
            # Perda adversarial do discriminador
            d_loss_real = self.gan_loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = self.gan_loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss_batch = (d_loss_real + d_loss_fake) / 2
            d_loss_val += d_loss_batch
            # Perda adversarial para o gerador (Least-Squares GAN)
            g_gan_loss = self.gan_loss_fn(tf.ones_like(fake_output), fake_output)
            # Perda de distorção
            g_distortion_loss = self.distortion_loss_fn(real_images, generated_images)
            # Perda total ponderada
            g_loss_batch = g_gan_loss + self.lambda_d * g_distortion_loss
            g_loss_val+= g_loss_batch
        d_loss_avg = d_loss_val / len(dataloader_val)
        g_loss_avg = g_loss_val / len(dataloader_val)
        return d_loss_avg, g_loss_avg

    def train_step(self, dataloader_train):
        d_loss = 0.
        g_loss = 0.
        for data in tqdm(dataloader_train):
            real_images = normalize_input(data)
            # Train the discriminator (max GAN loss)
            with tf.GradientTape() as tape:
                generated_images, quantized_latent = self.autoencoder(real_images)
                real_output = self.discriminator(real_images)
                fake_output = self.discriminator(generated_images)
                # Perda adversarial do discriminador
                d_loss_real = self.gan_loss_fn(tf.ones_like(real_output), real_output)
                d_loss_fake = self.gan_loss_fn(tf.zeros_like(fake_output), fake_output)
                d_loss_batch = (d_loss_real + d_loss_fake) / 2
                d_loss += d_loss_batch
                grads = tape.gradient(d_loss_batch, self.discriminator.trainable_weights)
                self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
            # Train the autoencoder (min GAN loss, distortion loss, entropy loss)
            with tf.GradientTape() as tape:
                generated_images, quantized_latent = self.autoencoder(real_images)
                fake_output = self.discriminator(generated_images)
                # Perda adversarial para o gerador (Least-Squares GAN)
                g_gan_loss = self.gan_loss_fn(tf.ones_like(fake_output), fake_output)
                # Perda de distorção
                g_distortion_loss = self.distortion_loss_fn(real_images, generated_images)
                # Perda total ponderada
                g_loss_batch = g_gan_loss + self.lambda_d * g_distortion_loss
                g_loss += g_loss_batch
                grads = tape.gradient(g_loss_batch, self.autoencoder.trainable_weights)
                self.g_optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_weights))
        d_loss_avg = d_loss / len(dataloader_train)
        g_loss_avg = g_loss / len(dataloader_train)
        return d_loss, g_loss
