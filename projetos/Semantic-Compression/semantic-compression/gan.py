import torch
from torch import nn
from tqdm import tqdm


class Block(nn.Module):
    """"""
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=1, padding='same', bias=False),
            nn.InstanceNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=1, padding='same', bias=False),
            nn.InstanceNorm2d(filters),
        )

    def forward(self, z):
        out = z + self.block(z)
        return out


class ResidualBlock(nn.Module):
    """"""
    def __init__(self, filters, kernel_size, n_blocks=3):
        super().__init__()
        self.blocks = nn.Sequential(*(Block(filters, kernel_size) for _ in range(n_blocks)))

    def forward(self, z):
        out = z + self.blocks(z)
        return out


class Encoder(nn.Module):
    """
    input: k7s1 [filters / 2**n_convs]
    downsample: k5s2 [filters / 2**(n_convs-i)], i=1,2,...,n_convs
    output: k3s1 [channels]
    """
    def __init__(self, filters, in_channels, out_channels, n_convs=4):
        super().__init__()
        self.h_in = nn.Sequential(
            nn.Conv2d(
                in_channels, filters // 2**n_convs,
                kernel_size=7, stride=1, padding=3, bias=False,
            ),
            nn.InstanceNorm2d(filters // 2**n_convs),
            nn.ReLU(),
        )
        self.downsample = nn.Sequential()
        for i in range(n_convs):
            self.downsample.append(
                nn.Conv2d(
                    filters // 2**(n_convs - i), filters // 2**(n_convs - i - 1),
                    kernel_size=5, stride=2, padding=2, bias=False,
                )
            )
            self.downsample.append(nn.InstanceNorm2d(filters // (2**(n_convs - i - 1))))
            self.downsample.append(nn.ReLU())
        self.h_out = nn.Sequential(
            nn.Conv2d(filters, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x):
        h = self.h_in(x)
        h_down = self.downsample(h)
        w = self.h_out(h_down)
        return w


class Quantizer(nn.Module):
    """"""
    def __init__(self, sigma, levels, l_min, l_max):
        super().__init__()
        self.sigma = sigma
        self.levels = levels
        self.l_min = l_min
        self.l_max = l_max

    def forward(self, z):
        B = z.shape[0]
        C = z.shape[1]
        z_vec = z.view(B, C, -1, 1)
        centers = torch.linspace(self.l_min, self.l_max, self.levels, dtype=z_vec.dtype, device=z_vec.device)
        dist = torch.square(torch.abs(z_vec - centers))
        phi = torch.softmax(-self.sigma * dist, axis=-1)
        z_hat = torch.sum(phi * centers, axis=-1).view(z.shape)
        return z_hat


class Generator(nn.Module):
    """
    input: k3s1 [filters]
    resnet: n_blocks x (k3s1 [filters] + k3s1 [filters])
    upsample: k5s2 [filters / 2**i], i=1,2,...,n_convs
    output: k7s1 [3]
    """
    def __init__(self, filters, n_blocks, in_channels, n_convs=4):
        super().__init__()
        self.h_in = nn.Sequential(
            nn.ConvTranspose2d(in_channels, filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*(ResidualBlock(filters, kernel_size=3) for _ in range(n_blocks)))
        self.upsample = nn.Sequential()
        for i in range(n_convs):
            self.upsample.append(
                nn.ConvTranspose2d(
                    filters // 2**i, filters // 2**(i + 1),
                    kernel_size=5, stride=2, padding=2, output_padding=1, bias=False,
                )
            )
            self.upsample.append(nn.InstanceNorm2d(filters // 2**(i + 1)))
            self.upsample.append(nn.ReLU())
        self.h_out = nn.Sequential(
            nn.ConvTranspose2d(filters // 2**n_convs, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
        )

    def forward(self, w_hat):
        z = self.h_in(w_hat)
        h = self.res_blocks(z)
        h_up = self.upsample(h)
        x_hat = self.h_out(h_up)
        return x_hat


class Discriminator(nn.Module):
    """
    downsample: k4s2 [filters / 2**(n_convs-i)], i=1,2,...,n_convs
    output: k4s1 [1] + dense(sigmoid)
    """
    def __init__(self, filters, n_convs=4, dropout=0.0):
        super().__init__()
        self.h_in = nn.Sequential(
            nn.Conv2d(
                4, filters // 2**n_convs,
                kernel_size=7, stride=1, padding=3, bias=False,
            ),
            nn.LeakyReLU(0.2),
        )
        self.downsample = nn.Sequential()
        for i in range(n_convs):
            self.downsample.append(
                nn.Conv2d(
                    filters // 2**(n_convs - i), filters // 2**(n_convs - i - 1),
                    kernel_size=5, stride=2, padding=2, bias=False,
                )
            )
            self.downsample.append(nn.InstanceNorm2d(filters // 2**(n_convs - i - 1)))
            self.downsample.append(nn.LeakyReLU(0.2))
            self.downsample.append(nn.Dropout(dropout))
        self.h_out = nn.Sequential(
            nn.Conv2d(filters, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, s=None):
        if s is None:
            s = torch.zeros_like(x[..., :1, :, :])
        xs = torch.concatenate([x, s], axis=-3)
        h = self.h_in(xs)
        h_down = self.downsample(h)
        y = self.h_out(h_down)
        return y


class AutoEncoder(nn.Module):
    def __init__(self, filters, n_blocks, channels, n_convs, sigma, levels, l_min, l_max):
        super().__init__()
        self.encoder = Encoder(filters, 4, channels, n_convs)
        self.quantizer = Quantizer(sigma, levels, l_min, l_max)
        self.decoder = Generator(filters, n_blocks, 2*channels, n_convs)
        self.extractor = Encoder(filters, 1, channels, n_convs)

    def forward(self, x, s=None):
        if s is None:
            s = torch.zeros_like(x[..., :1, :, :])
        xs = torch.concatenate([x, s], axis=-3)
        w = self.encoder(xs)
        w_hat = self.quantizer(w)
        v = self.extractor(s)
        v_hat = self.quantizer(v)
        wv_hat = torch.concatenate([w_hat, v_hat], axis=-3)
        x_hat = self.decoder(wv_hat)
        return x_hat


def weights_init(m):
    """inicialização sugerida para DCGANs (Radford et al. 2015)"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

class CGAN(nn.Module):
    def __init__(self, filters, n_blocks, channels, n_convs, sigma=1000, levels=5, l_min=-2, l_max=2,
                 lambda_d=10, gan_loss="BCE", betas=(0.5, 0.999), ae_lr=3e-4, dc_lr=3e-4, device=None,
                 dropout=0.0, real_label=1.0, fake_label=0.0, input_noise=0.0, c_mode=0, run_id=None):
        super().__init__()
        self.lambda_d = lambda_d
        self.input_noise = input_noise
        self.real_label = real_label
        self.fake_label = fake_label
        self.device = device
        self.c_mode = c_mode
        self.autoencoder = AutoEncoder(filters, n_blocks, channels, n_convs, sigma, levels, l_min, l_max).to(device)
        self.discriminator = Discriminator(filters, n_convs, dropout).to(device)
        self.autoencoder.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.pre_opt = torch.optim.AdamW(self.autoencoder.parameters(), lr=ae_lr, betas=betas)
        self.ae_opt = torch.optim.AdamW(self.autoencoder.parameters(), lr=ae_lr, betas=betas)
        self.dc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=dc_lr, betas=betas)
        self.perception_loss = getattr(nn, f"{gan_loss}Loss")()
        self.distortion_loss = nn.MSELoss()
        self.filename = run_id

    def compress(self, x, s=None):
        if s is None:
            s = torch.zeros_like(x[..., :1, :, :])
        xs = torch.concatenate([x, s], axis=-3)
        w = self.autoencoder.encoder(xs)
        w_hat = self.autoencoder.quantizer(w)
        v = self.autoencoder.extractor(s)
        v_hat = self.autoencoder.quantizer(v)
        return w_hat, v_hat

    def decompress(self, w_hat, v_hat=None):
        if not v_hat:
            v_hat = torch.zeros_like(w_hat)
        wv_hat = torch.concatenate([w_hat, v_hat], axis=-3)
        x_hat = self.autoencoder.generator(wv_hat)
        return x_hat

    def compute_perception(self, image, mask, label):
        # with torch.autocast(device_type=self.device):
        if self.c_mode > 0:
            out = self.discriminator(image, mask)
        else:
            out = self.discriminator(image)
        label = torch.full(out.shape, label, device=out.device)
        err = self.perception_loss(out, label)
        err.backward()
        return err.item()

    def compute_distortion(self, x, s, x_hat, label):
        # with torch.autocast(device_type=self.device):
        if self.c_mode > 0:
            out = self.discriminator(x_hat, s)
        else:
            out = self.discriminator(x_hat)
        label = torch.full(out.shape, label, device=out.device)
        err = self.lambda_d * self.distortion_loss(x_hat, x) + self.perception_loss(out, label)
        err.backward()
        return err.item()        

    def train_step(self, batch):
        x_batch = batch['x'].to(self.device)
        s_batch = batch['s'].to(self.device)
        noise = self.input_noise * torch.randn(x_batch.shape, device=self.device, dtype=x_batch.dtype)
        x0_batch = x_batch + noise
        # treinamento discriminador
        for param in self.discriminator.parameters():
            param.grad = None
        dc_real = self.compute_perception(x0_batch, s_batch, self.real_label)
        if self.c_mode == 2:
            x_hat = self.autoencoder(x_batch, s_batch)
        else:
            x_hat = self.autoencoder(x_batch)
        x0_hat = x_hat + noise
        dc_fake = self.compute_perception(x0_hat.detach(), s_batch, self.fake_label)
        self.dc_opt.step()
        # treinamento autoencoder
        for param in self.autoencoder.parameters():
            param.grad = None
        ae_loss = self.compute_distortion(x0_batch, s_batch, x0_hat, self.real_label)
        self.ae_opt.step()
        # loss total
        dc_loss = dc_real + dc_fake
        return ae_loss, dc_loss

    def train_epoch(self, dataloader):
        ae_avg = 0.0
        dc_avg = 0.0
        # pbar = tqdm(dataloader)
        for batch in dataloader:
            ae_loss, dc_loss = self.train_step(batch)
            # pbar.set_description(f"AE: {ae_loss:.6f}, D: {dc_loss:.6f}")
            ae_avg += ae_loss
            dc_avg += dc_loss
        torch.cuda.empty_cache()
        return ae_avg / len(dataloader), dc_avg / len(dataloader)

    def pre_train_step(self, batch):
        x_batch = batch['x'].to(self.device)
        s_batch = batch['s'].to(self.device)
        # with torch.autocast(device_type=self.device):
        if self.c_mode == 2:
            x_hat = self.autoencoder(x_batch, s_batch)
        else:
            x_hat = self.autoencoder(x_batch)
        for param in self.autoencoder.parameters():
            param.grad = None
        err = self.distortion_loss(x_hat, x_batch)
        err.backward()
        loss = err.item()
        self.pre_opt.step()
        return loss

    def pre_train_epoch(self, dataloader):
        avg_loss = 0.0
        # pbar = tqdm(dataloader)
        for batch in dataloader:
            loss = self.pre_train_step(batch)
            # pbar.set_description(f"AE: {loss:.6f}")
            avg_loss += loss
        torch.cuda.empty_cache()
        return avg_loss / len(dataloader)

    def save_models(self, epoch, model_path):
        torch.save(self.autoencoder.state_dict(), f"{model_path}/ae_{self.filename}_{epoch}.pth")
        torch.save(self.discriminator.state_dict(), f"{model_path}/dc_{self.filename}_{epoch}.pth")
