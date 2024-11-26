import os

import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import v2
from tqdm.auto import tqdm

from config import *
from dataset import SemanticData, SemanticResize
from gan import CGAN
from plots import plot_reconstruction
from predictTrain import predict


def pre_train(model, dataloader):
    losses = []
    pbar = tqdm(range(PT_EPOCHS))
    for epoch in pbar:
        loss = model.pre_train_epoch(dataloader)
        losses.append(loss)
        pbar.set_description(f"AE: {loss:.6f}")
        if (epoch+1) % PLOT_EVERY == 0:
            plot_reconstruction(model, dataloader, epoch+1, PATH_FIGS, model.filename)
            model.train()
        if (epoch+1) % SAVE_EVERY == 0:
            model.save_models(epoch+1, PATH_MODELS)
    return model, losses


def fine_tune(model, dataloader, dataloader2):
    ae_losses = []
    dc_losses = []
    
    pbar = tqdm(range(FT_EPOCHS))
    for epoch in pbar:
        ae_loss, dc_loss = model.train_epoch(dataloader)
        ae_losses.append(ae_loss)
        dc_losses.append(dc_loss)
        pbar.set_description(f"AE: {ae_loss:.6f}, D: {dc_loss:.6f}")
        if (epoch+1) % PLOT_EVERY == 0:
            plot_reconstruction(model, dataloader, PT_EPOCHS+epoch+1, PATH_FIGS, model.filename)
            model.train()
        if (epoch+1) % SAVE_EVERY == 0:
            model.save_models(PT_EPOCHS+epoch+1, PATH_MODELS)
            predict(dataloader2, epoch+1)
    return model, ae_losses, dc_losses


def main():
    """
    """
    if not os.path.exists(PATH_MODELS):
        os.mkdir(PATH_MODELS)
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    train_set = SemanticData(
        PATH_TRAIN, CROP_SHAPE,
        transform=v2.Compose([
            v2.ToImage(),
            SemanticResize(size=SHORT_SIZE),
            v2.RandomHorizontalFlip(p=0.5),
        ]),
        x_transform=v2.Compose([
            v2.ColorJitter(brightness=.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,), (0.5,)),
        ]),
        s_transform=v2.Compose([
            v2.ToDtype(torch.uint8),
        ]),
    )
    train_loader = data.DataLoader(train_set,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=True)

    test_set = SemanticData(
        PATH_TEST, CROP_SHAPE,
        transform=v2.Compose([
            v2.ToImage(),
            SemanticResize(size=SHORT_SIZE),
        ]),
        x_transform=v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,), (0.5,)),
        ]),
        s_transform=v2.Compose([
            v2.ToDtype(torch.uint8),
        ]),
    )
    test_loader = data.DataLoader(test_set,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)

    model = CGAN(FILTERS, N_BLOCKS, CHANNELS, N_CONVS, SIGMA, LEVELS, L_MIN, L_MAX,
                 LAMBDA_D, GAN_LOSS, OPTIMIZER_BETAS, AE_LR, DC_LR, DEVICE,
                 DROPOUT, REAL_LABEL, FAKE_LABEL, INPUT_NOISE, C_MODE, RUN_ID)
    model = model.to(DEVICE)
    if PT_EPOCHS > 0:
        model, losses = pre_train(model, train_loader)
        np.savetxt(os.path.join(PATH_MODELS, "pt_" + model.filename + ".txt"), np.array(losses))
        print(losses[-1])
    model, ae_losses, dc_losses = fine_tune(model, train_loader, test_loader)
    np.savetxt(os.path.join(PATH_MODELS, "ft_" + model.filename + ".txt"), np.array([ae_losses, dc_losses]))

if __name__ == '__main__':
    main()
