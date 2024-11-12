import os

import matplotlib.pyplot as plt
import numpy as np


def plot_samples(images, fig_path, fname):
    plt.figure(figsize=(10, 10))
    for j in range(16):
        img = (images[j].cpu().detach() + 1) * 0.5
        img = img.permute(1, 2, 0).numpy()
        plt.subplot(4, 4, j+1)
        plt.imshow(img)
        plt.axis('off')
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    plt.savefig(f'{fig_path}/{fname}.jpg')

def plot_reconstruction(autoencoder, dataloader, epoch, fig_path, filename):
    indices = np.random.randint(len(dataloader), size=16)
    sampled_images = [dataloader.dataset[i][0].to(device) for i in indices]
    autoencoder.eval()
    output_images = [autoencoder(img) for img in sampled_images]
    fname = f"og_{filename}_{epoch}"
    plot_samples(sampled_images, fig_path, fname)
    fname = f"re_{filename}_{epoch}"
    plot_samples(output_images, fig_path, fname)

def make_plot(model, losses, test_batch):
    xhat, what = model(test_batch)
    fig = plt.figure(figsize=(12, 6))

    gs = fig.add_gridspec(2, 4, width_ratios=[2, 2, 2, 2]) 
    ax1 = fig.add_subplot(gs[0, 0:2]) 
    # ax1.set_ylim(-1, 2)
    ax1.plot(losses[0], label='Loss_train')
    ax1.plot(losses[2], label='Loss_val')
    ax1.set_title('D_losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Valor')
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0:2])  
    # ax2.set_ylim(0, 12)
    ax2.plot(losses[1], label='Loss_train')
    ax2.plot(losses[3], label='Loss_val')
    ax2.set_title('G_losses')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Valor')
    ax2.legend()

    ax_img1 = fig.add_subplot(gs[0, 2:4]) 
    ax_img1.imshow(np.uint8(test_batch[0]))
    ax_img1.axis('off')  

    ax_img2 = fig.add_subplot(gs[1, 2:4])  
    ax_img2.imshow(np.uint8(xhat[0]))
    ax_img2.axis('off') 

    plt.tight_layout()
    plt.show()
