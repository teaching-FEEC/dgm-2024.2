import sys
import os
sys.path.append(os.path.join(os.getcwd(), ".."))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from tqdm import tqdm
import wandb

from satellite2map.datasets import Maps
from satellite2map.models.generator import UnetGenerator
from satellite2map.models.discriminator import ConditionalDiscriminator
from satellite2map.losses import GeneratorLoss, DiscriminatorLoss
import satellite2map.transforms as T
from satellite2map.metricas import ssim_metric, psnr_metric, mse_metric

from dataclasses import dataclass
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Implementação do Pix2Pix"""
@dataclass
class hyperparameters:
    # training hyperparams
    n_epochs: int = 100
    batch_size: int = 32
    lr: float = 2e-3

hyperparams = hyperparameters()


""" Carregando os dados """
transforms = T.Compose([T.Resize((256,256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])])

train_dataset = Maps(root='../data/raw', mode='train', transform=transforms, download=False)
val_dataset   = Maps(root='../data/raw', mode='val', transform=transforms, download=False)

train_dataloader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True)
val_dataloader   = DataLoader(val_dataset,   batch_size=hyperparams.batch_size, shuffle=True)

import matplotlib.pyplot as plt
import numpy as np

for i, (sat, map) in enumerate(train_dataset):
    i = i + 1
    print(i, sat.shape, map.shape)

    ax = plt.subplot(1, 4, i)
    plt.tight_layout()
    #ax.set_title('Satellite #{}'.format(i))
    ax.axis('off')
    sat = sat * 0.5 + 0.5
    plt.imshow(sat.permute((1, 2, 0)).numpy())

    ax = plt.subplot(2, 4, i)
    plt.tight_layout()
    #ax.set_title('Map #{}'.format(i))
    ax.axis('off')
    map = map * 0.5 + 0.5
    plt.imshow(map.permute((1, 2, 0)).numpy())

    if i == 4:
        plt.show()
        break


""" Preparando para o treinamento """
generator = UnetGenerator().to(DEVICE)
discriminator = ConditionalDiscriminator().to(DEVICE)

g_criterion = GeneratorLoss()
d_criterion = DiscriminatorLoss()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=hyperparams.lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=hyperparams.lr, betas=(0.5, 0.999))

# Lista para armazenar as métricas
metrics_history = []

""" Loop de Treinamento """
for epoch in range(hyperparams.n_epochs):

    # Variáveis para monitoramento das métricas
    mse_epoch = 0
    psnr_epoch = 0
    ssim_epoch = 0
    num_batches = 0

    print(f'[Epoch {epoch + 1}/{hyperparams.n_epochs}]')
    # training
    generator.train()
    discriminator.train()
    train_loss_g = 0.
    train_loss_d = 0.
    start = time.time()
    tqdm.write('Training')
    for x, real in tqdm(train_dataloader):
        x = x.to(DEVICE)
        real = real.to(DEVICE)

        # Generator`s loss
        fake = generator(x)
        fake_pred = discriminator(fake, x)
        g_loss = g_criterion(fake, real, fake_pred)

        # Discriminator`s loss
        fake = generator(x).detach()
        fake_pred = discriminator(fake, x)
        real_pred = discriminator(real, x)
        d_loss = d_criterion(fake_pred, real_pred)

        # Generator`s params update
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Discriminator`s params update
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # add batch losses
        train_loss_g += g_loss.item()
        train_loss_d += d_loss.item()

    # obtain per batch losses
    train_loss_g = train_loss_g / len(train_dataloader)
    train_loss_d = train_loss_d / len(train_dataloader)
    # count timeframe
    end = time.time()
    tm = (end - start)

    # validation
    generator.eval()
    discriminator.eval()
    val_loss_g = 0.
    val_loss_d = 0.
    tqdm.write('Validation')
    for x, real in tqdm(val_dataloader):
        x = x.to(DEVICE)
        real = real.to(DEVICE)

        with torch.no_grad():
            # Generator`s loss
            fake = generator(x)
            fake_pred = discriminator(fake, x)
            g_loss = g_criterion(fake, real, fake_pred)

            # Discriminator`s loss
            fake = generator(x).detach()
            fake_pred = discriminator(fake, x)
            real_pred = discriminator(real, x)
            d_loss = d_criterion(fake_pred, real_pred)

        # add batch losses
        val_loss_g += g_loss.item()
        val_loss_d += d_loss.item()

        # Calcular as métricas de avaliação todo
        mse_value = mse_metric(x, real)
        psnr_value = psnr_metric(x, real)
        ssim_value = ssim_metric(x, real)

        # Acumular as métricas
        mse_epoch += mse_value
        psnr_epoch += psnr_value
        ssim_epoch += ssim_value
        num_batches += 1

    # Média das métricas por época
    mse_epoch /= num_batches
    psnr_epoch /= num_batches
    ssim_epoch /= num_batches

    # Adicionar os valores da época ao histórico
    metrics_history.append({
        "epoch": epoch + 1,
        "mse": mse_epoch,
        "psnr": psnr_epoch,
        "ssim": ssim_epoch
    })

    #print(f"Epoch [{epoch + 1}/{num_epochs}]: MSE: {mse_epoch:.4f}, PSNR: {psnr_epoch:.2f}, SSIM: {ssim_epoch:.4f}")

    # obtain per batch losses
    val_loss_g = val_loss_g / len(val_dataloader)
    val_loss_d = val_loss_d / len(val_dataloader)


    # logging
    """wandb.log({"generator_train_loss": train_loss_g,
               "generator_val_loss": val_loss_g,
               "discriminator_train_loss": train_loss_d,
               "discriminator_val_loss": val_loss_d,
               #    "time": tm
               })
    print(f"[G loss: {g_loss}] [D loss: {d_loss}] ETA: {tm}")"""

    if epoch % 10 != 0:
        continue

    # saving the model every 10 epochs
    torch.save({'gen_weights': generator.state_dict(),
                'disc_weights': discriminator.state_dict()},
               f'../models/pix2pix/checkpoints/checkpoint_{epoch}.pth')

    # visualizing the results
    fig = plt.figure(figsize=(10, 10))
    random_batch = val_dataloader[torch.randint(0, len(val_dataloader), 4)]
    for i, (sat, map) in enumerate(random_batch):
        i = i + 1
        gen_map = generator(sat.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

        ax = fig.add_subplot(3, 4, i)
        ax.set_title('Satellite #{}'.format(i))
        ax.axis('off')
        sat = sat * 0.5 + 0.5
        plt.imshow(sat.permute((1, 2, 0)).numpy())

        ax = fig.add_subplot(3, 4, i + 4)
        ax.set_title('Map #{}'.format(i))
        ax.axis('off')
        print(f'maps: {map.max()}, {map.min()}')
        map = map * 0.5 + 0.5
        plt.imshow(map.permute((1, 2, 0)).numpy())

        ax = fig.add_subplot(3, 4, i + 8)
        ax.set_title('Generated map #{}'.format(i))
        ax.axis('off')
        print(f'gen maps: {gen_map.max()}, {gen_map.min()}')
        gen_map = gen_map * 0.5 + 0.5
        plt.imshow(gen_map.permute((1, 2, 0)).cpu().detach().numpy())
        if i == 4:
            fig.suptitle(f'Epoch {epoch + 1}')
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            plt.show()
            break

import pandas as pd

# Salvar o histórico das métricas em um CSV após todas as épocas
df = pd.DataFrame(metrics_history)
df.to_csv('../satellite2map/models/metrics_history.csv', index=False)

""" Inferência """
gen_state_dict = torch.load('../models/pix2pix/checkpoints/checkpoint_100.pth')['gen_weights']
generator.load_state_dict(gen_state_dict)

fig = plt.figure(figsize=(8, 8))
indexes = torch.randint(0, len(val_dataset), (4,))
for i in range(4):
    (sat, map) = val_dataset[indexes[i]]
    i = i + 1
    # print(i, sat.shape, map.shape)
    gen_map = generator(sat.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

    ax = fig.add_subplot(3, 4, i)
    ax.set_title('Satellite #{}'.format(i))
    ax.axis('off')
    sat = sat * 0.5 + 0.5
    plt.imshow(sat.permute((1, 2, 0)).numpy())

    ax = fig.add_subplot(3, 4, i+4)
    ax.set_title('Map #{}'.format(i))
    ax.axis('off')
    print(f'maps: {map.max()}, {map.min()}')
    map = map * 0.5 + 0.5
    plt.imshow(map.permute((1, 2, 0)).numpy())

    ax = fig.add_subplot(3, 4, i+8)
    ax.set_title('Generated map #{}'.format(i))
    ax.axis('off')
    print(f'gen maps: {gen_map.max()}, {gen_map.min()}')
    gen_map = gen_map * 0.5 + 0.5
    plt.imshow(gen_map.permute((1, 2, 0)).cpu().detach().numpy())
    if i == 4:
        fig.tight_layout()
        fig.suptitle('Epoch 100')
        plt.show()
        break


""" saving the model every 10 epochs """
torch.save({'gen_weights': generator.state_dict(),
            'disc_weights': discriminator.state_dict()},
            f'../models/pix2pix/checkpoints/checkpoint_100.pth')