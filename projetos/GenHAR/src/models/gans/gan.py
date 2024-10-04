# Modelos Gerativos (GAN, VAE, CGAN)
class GAN:
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.latent_dim),
            layers.Dense(self.latent_dim, activation='tanh')
        ])
        return model

    def build_discriminator(self):
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.latent_dim),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def build_gan(self):
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy')
        self.discriminator.trainable = False
        gan = models.Sequential([self.generator, self.discriminator])
        gan.compile(optimizer='adam', loss='binary_crossentropy')
        return gan

    def train(self, data, epochs=5000, batch_size=64):
        for epoch in range(epochs):
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            generated_data = self.generator.predict(noise)
            real_data = data[np.random.randint(0, data.shape[0], batch_size)]
            labels_real = np.ones((batch_size, 1))
            labels_fake = np.zeros((batch_size, 1))
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_data, labels_real)
            d_loss_fake = self.discriminator.train_on_batch(generated_data, labels_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, labels_real)
            
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}, G Loss: {g_loss}')

