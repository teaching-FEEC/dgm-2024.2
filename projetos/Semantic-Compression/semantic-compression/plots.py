import os

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def plot_reconstruction(model, dataloader, epoch, fig_path, filename):
    sampled_images = [dataloader.dataset[i]['x'].to(model.device) for i in range(8)]
    model.eval()
    with torch.no_grad():
        output_images = [model.autoencoder(img) for img in sampled_images]
    images = sampled_images + output_images
    grid = (make_grid(images, nrow=4) + 1) * 0.5
    plt.figure(figsize=(16, 16))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.savefig(f'{fig_path}/{filename}_{epoch}.jpg', bbox_inches='tight')

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
