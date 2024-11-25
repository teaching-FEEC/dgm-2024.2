import sys
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from functools import partial
import numpy as np
from tqdm.auto import tqdm
from utils.dataset_utils import split_axis_reshape, dict_class_samples
from models.diffusion.unet.unet_1d import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# Gaussian diffusion trainer class
# Code Source: https://github.com/imics-lab/biodiffusion/blob/main/src/ddpm.py


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    """
    Generate a warm-up schedule for betas during training.

    Args:
        linear_start (float): Initial value for beta.
        linear_end (float): Final value for beta.
        n_timestep (int): Total number of timesteps.
        warmup_frac (float): Fraction of timesteps for warm-up.

    Returns:
        numpy.ndarray: Array of beta values.
    """
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """
    Generate beta schedule based on the specified schedule type.

    Args:
        schedule (str): Type of schedule.
        n_timestep (int): Total number of timesteps.
        linear_start (float): Initial value for beta (default: 1e-4).
        linear_end (float): Final value for beta (default: 2e-2).
        cosine_s (float): S parameter for cosine schedule (default: 8e-3).

    Returns:
        numpy.ndarray: Array of beta values.
    """
    if schedule == "quad":
        betas = np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "jsd":
        betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# Define the Diffusion class
class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, seq_length, channels=3, loss_type="l1", schedule_opt=None):
        """
        Gaussian Diffusion Probabilistic Model.

        Args:
            denoise_fn (callable): Denoising function.
            seq_length (int): Length of the input sequence.
            channels (int): Number of channels in the input sequence (default: 3).
            loss_type (str): Type of loss ('l1' or 'l2') (default: 'l1').
            schedule_opt (dict): Options for the diffusion schedule.
        """
        super().__init__()
        self.channels = channels
        self.seq_length = seq_length
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        if schedule_opt is not None:
            self.set_new_noise_schedule(schedule_opt, device)
        self.set_loss(device)

    def set_loss(self, device):
        """
        Set the loss function based on the specified loss type.

        Args:
            device (str): Device to use for the loss function.
        """
        if self.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="sum").to(device)
        elif self.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="sum").to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        """
        Set a new noise schedule for the diffusion model.

        Args:
            schedule_opt (dict): Options for the diffusion schedule.
            device (str): Device to use for the schedule.
        """
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt["schedule"],
            n_timestep=schedule_opt["n_timestep"],
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict the starting point from noise.

        Args:
            x_t (torch.Tensor): Current point in the diffusion process.
            t (int): Current timestep.
            noise (torch.Tensor): Noise tensor.

        Returns:
            torch.Tensor: Predicted starting point.
        """
        return (
            self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Calculate the posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_start (torch.Tensor): Starting point in the diffusion process.
            x_t (torch.Tensor): Current point in the diffusion process.
            t (int): Current timestep.

        Returns:
            tuple: Tuple containing the posterior mean and log variance.
        """
        posterior_mean = (
            self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        )
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        """
        Calculate the mean and log variance for p(x_t | x_{t-1}, x_0).

        Args:
            x (torch.Tensor): Current point in the diffusion process.
            t (int): Current timestep.
            clip_denoised (bool): Whether to clip the denoised signal.
            condition_x (torch.Tensor): Conditional input.

        Returns:
            tuple: Tuple containing the model mean and posterior log variance.
        """
        batch_size = x.shape[0]
        noise_level = (
            torch.tensor([self.sqrt_alphas_cumprod_prev[t + 1]])
            .repeat(batch_size, 1)
            .to(x.device)
            .float()
        )
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level, condition_x)
            )
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        """
        Generate a sample from p(x_t | x_{t-1}, x_0).

        Args:
            x (torch.Tensor): Current point in the diffusion process.
            t (int): Current timestep.
            clip_denoised (bool): Whether to clip the denoised signal.
            condition_x (torch.Tensor): Conditional input.

        Returns:
            torch.Tensor: Sampled point.
        """
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x
        )
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, cond=None):
        """
        Generate a sample from the diffusion process.

        Args:
            x_in (tuple): Input shape tuple.
            continous (bool): Whether to return continuous samples.

        Returns:
            torch.Tensor: Sampled sequence.
        """
        device = self.betas.device
        sample_inter = self.num_timesteps // 10
        shape = x_in
        img = torch.randn(shape, device=device)
        ret_img = img.unsqueeze(0)
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample(img, i, condition_x=cond)
            if (i + 1) % sample_inter == 0:
                ret_img = torch.cat([ret_img, img.unsqueeze(0)], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False, cond=None):
        """
        Generate samples from the diffusion process.

        Args:
            batch_size (int): Batch size.
            continous (bool): Whether to return continuous samples.

        Returns:
            torch.Tensor: Generated samples.
        """
        seq_length = self.seq_length
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, seq_length), continous, cond)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        """
        Generate super-resolution samples from the diffusion process.

        Args:
            x_in (tuple): Input shape tuple.
            continous (bool): Whether to return continuous samples.

        Returns:
            torch.Tensor: Generated super-resolution samples.
        """
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        """
        Sample from the diffusion process q(x_t | x_{t-1}).

        Args:
            x_start (torch.Tensor): Starting point in the diffusion process.
            continuous_sqrt_alpha_cumprod (torch.Tensor): Continuous square root of alpha cumprod.
            noise (torch.Tensor): Noise tensor.

        Returns:
            torch.Tensor: Sampled point.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            continuous_sqrt_alpha_cumprod * x_start
            + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, cond=None, noise=None):
        """
        Calculate losses for p(x_t | x_{t-1}).

        Args:
            x_in (dict): Dictionary containing input tensors.
            noise (torch.Tensor): Noise tensor.

        Returns:
            torch.Tensor: Loss value.
        """
        x_start = x_in
        [b, c, lenth] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = (
            torch.tensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[t - 1], self.sqrt_alphas_cumprod_prev[t], size=b
                )
            )
            .to(x_start.device)
            .float()
        )
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1),
            noise=noise,
        )

        x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod, cond)

        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, cond=None, *args, **kwargs):
        """
        Forward pass of the diffusion model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Loss value.
        """
        return self.p_losses(x, cond, *args, **kwargs)


class DiffusionGenerator:

    def __init__(self, config):
        self.config = config
        self.model = None

    def train(self, X_train, y_train):
        self.scaler = {}
        self.losses = {}
        if self.config["is_conditional"]:
            self.train_conditional(X_train, y_train)
        else:
            self.train_separated(X_train, y_train)
        return self.model, self.losses

    def train_separated(self, X_train, y_train):
        params = self.config["parameters"]

        x_df = X_train.copy()
        self.columns_names = x_df.columns

        self.model = {}
        x_data = split_axis_reshape(x_df)
        class_data_dict = dict_class_samples(x_data, y_train.copy())
        for class_label in class_data_dict.keys():
            train_data = class_data_dict[class_label]
            assert train_data.shape[-2:] == (6, 60)

            transform_data = train_data.transpose(0, 2, 1).reshape(-1, 6)
            scaler = StandardScaler()
            scaler.fit(transform_data)
            transform_data = scaler.transform(transform_data)
            train_data = transform_data.reshape(train_data.shape[0], 60, 6).transpose(0, 2, 1)
            train_data = torch.from_numpy(train_data)

            model = UNet(
                in_channel=params["in_channel"],
                out_channel=params["out_channel"],
                seq_length=60,
            ).to(device)
            model, loss_hist = self.train_model(model, train_data)
            self.losses[class_label] = loss_hist
            self.model[class_label] = model
            self.scaler[class_label] = scaler

    def train_conditional(self, X_train, y_train):
        params = self.config["parameters"]
        x_df = X_train.copy()
        self.columns_names = x_df.columns
        self.classes = np.unique(y_train.values)

        X_data = []
        Y_data = []
        split_data = split_axis_reshape(x_df)
        class_data_dict = dict_class_samples(split_data, y_train.copy())
        for class_label in class_data_dict.keys():
            train_data = class_data_dict[class_label]
            assert train_data.shape[-2:] == (6, 60)

            transform_data = train_data.transpose(0, 2, 1).reshape(-1, 6)
            scaler = StandardScaler()
            scaler.fit(transform_data)
            transform_data = scaler.transform(transform_data)
            transform_data = transform_data.reshape(train_data.shape[0], 60, 6).transpose(0, 2, 1)
            X_data.append(transform_data)
            Y_data.append(np.ones((train_data.shape[0], 1)) * class_label)
            self.scaler[class_label] = scaler

        X_data = np.concatenate(X_data, axis=0)
        Y_data = np.concatenate(Y_data, axis=0)
        X_data = torch.from_numpy(X_data)
        Y_data = torch.from_numpy(Y_data).int()
        model = UNet(
            in_channel=params["in_channel"],
            out_channel=params["out_channel"],
            seq_length=60,
            conditional=True,
        ).to(device)

        assert X_data.shape[0] == Y_data.shape[0]
        assert X_data.shape[-2:] == (6, 60)
        assert Y_data.shape[1] == 1
        self.model, loss_hist = self.train_model(model, X_data, Y_data)
        self.losses['all'] = loss_hist

    def generate(self, n_samples):
        if self.model is None:
            raise RuntimeError("The model has not yet been trained")

        if self.config["is_conditional"]:
            return self.gen_conditional(n_samples)
        else:
            return self.gen_separated(n_samples)

    def gen_conditional(self, n_samples):
        synthetic_df = pd.DataFrame()
        samples_per_class = n_samples // self.classes.shape[0]
        diffusion = GaussianDiffusion(
            self.model,
            60,
            channels=6,
            loss_type="l2",
            schedule_opt={"schedule": "cosine", "n_timestep": 1000, "cosine_s": 8e-3},
        )
        for class_label in self.scaler.keys():
            cond_batch = torch.ones(samples_per_class, 1) * class_label
            cond_batch = cond_batch.to(device).int()
            synthetic_samples = (
                diffusion.sample(batch_size=samples_per_class, cond=cond_batch).detach().cpu()
            )
            synthetic_samples = synthetic_samples.transpose(2, 1).reshape(-1, 6)
            synthetic_samples = self.scaler[class_label].inverse_transform(synthetic_samples)
            synthetic_samples = synthetic_samples.reshape(samples_per_class, 60, 6).transpose(
                0, 2, 1
            )
            synthetic_samples = synthetic_samples.reshape(samples_per_class, -1)
            class_df = pd.DataFrame(synthetic_samples, columns=self.columns_names)
            class_df["label"] = [class_label] * samples_per_class
            synthetic_df = pd.concat([synthetic_df, class_df], ignore_index=True)

        return synthetic_df

    def gen_separated(self, n_samples):
        synthetic_df = pd.DataFrame()
        samples_per_class = n_samples // len(self.model)
        for class_label, model in self.model.items():
            diffusion = GaussianDiffusion(
                model,
                60,
                channels=6,
                loss_type="l2",
                schedule_opt={"schedule": "cosine", "n_timestep": 1000, "cosine_s": 8e-3},
            )
            synthetic_samples = diffusion.sample(batch_size=samples_per_class).detach().cpu()
            synthetic_samples = synthetic_samples.transpose(2, 1).reshape(-1, 6)
            synthetic_samples = self.scaler[class_label].inverse_transform(synthetic_samples)
            synthetic_samples = synthetic_samples.reshape(samples_per_class, 60, 6).transpose(
                0, 2, 1
            )
            synthetic_samples = synthetic_samples.reshape(samples_per_class, -1)
            class_df = pd.DataFrame(synthetic_samples, columns=self.columns_names)
            class_df["label"] = [class_label] * samples_per_class
            synthetic_df = pd.concat([synthetic_df, class_df], ignore_index=True)

        return synthetic_df

    def train_model(self, model, x_train, y_train=None):
        diffusion_process = GaussianDiffusion(
            model,
            x_train.shape[1],
            channels=x_train.shape[-1],
            loss_type="l2",
            schedule_opt={"schedule": "cosine", "n_timestep": 1000, "cosine_s": 8e-3},
        )

        optimizer = optim.AdamW(model.parameters(), lr=3e-4)

        if y_train is not None:
            dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
        else:
            dataloader = DataLoader(TensorDataset(x_train), batch_size=64, shuffle=True)

        loss_hist = []
        for e in tqdm(range(self.config["epochs"])):
            epoch_loss = 0
            for batch in tqdm(dataloader):
                if y_train is not None:
                    batch_x = batch[0].to(device)
                    batch_y = batch[1].to(device)

                    loss = diffusion_process(batch_x.float(), cond=batch_y)
                else:
                    batch = batch[0].to(device)

                    loss = diffusion_process(batch.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()

            loss_hist.append(epoch_loss / len(dataloader))
        return model, loss_hist
