import os

import numpy as np
import pyiqa
import torch
from torch.utils import data
from torchvision.transforms import v2
from tqdm.auto import tqdm

from config import *
from dataset import SemanticData, SemanticResize
from gan import CGAN
from plots import plot_reconstruction
from segmentation import psp, deeplab

def iou(s, s_hat):
    labels = np.unique(s)
    iou = 0.0
    for label in labels:
        thresh1 = np.where(s == label, 255, 0).astype(np.uint8)
        thresh2 = np.where(s_hat == label, 255, 0).astype(np.uint8)
        intersection = cv2.bitwise_and(thresh1, thresh2)
        union = cv2.bitwise_or(thresh1, thresh2)
        label_iou = np.sum(intersection) / np.sum(union)
        iou += label_iou * np.sum(thresh1)
    return iou


def evaluate(model, seg, dataloader):
    ds = []
    psis = []
    ious = []
    d = pyiqa.create_metric("ms_ssim", device=DEVICE)
    psi = pyiqa.create_metric("musiq", device=DEVICE)
    for batch in tqdm(dataloader):
        x_test = batch['x'].to(DEVICE)
        s_test = batch['s'].to(DEVICE)
        with torch.no_grad():
            if C_MODE == 2:
                x_hat = model.autoencoder(x_test, s_test)
            else:
                x_hat = model.autoencoder(x_test)
            x_test = x_test * 0.5 + 0.5
            x_hat = x_hat * 0.5 + 0.5
            for i in range(len(x_hat)):
                s0 = s_test[i].permute(1, 2, 0).cpu().numpy()
                x = x_hat[i].permute(1, 2, 0).cpu().numpy()
                x = (x * 255).astype(np.uint8)
                s = phi(seg, x)
                ious.append(iou(s0, s))
            ds.append(d(x_test, x_hat))
            psis.append(psi(x_hat))
    return torch.concat(ds), torch.from_numpy(np.array(ious)), torch.concat(psis)

def main():
    # test dataset
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
    # load trained model
    model = CGAN(FILTERS, N_BLOCKS, CHANNELS, N_CONVS, SIGMA, LEVELS, L_MIN, L_MAX,
                 LAMBDA_D, GAN_LOSS, OPTIMIZER_BETAS, AE_LR, DC_LR, DEVICE,
                 DROPOUT, REAL_LABEL, FAKE_LABEL, INPUT_NOISE, C_MODE, RUN_ID)
    ae_checkpoint = torch.load(os.path.join(PATH_MODELS, f"ae_{RUN_ID}_{PT_EPOCHS+FT_EPOCHS}.pth"))
    dc_checkpoint = torch.load(os.path.join(PATH_MODELS, f"dc_{RUN_ID}_{PT_EPOCHS+FT_EPOCHS}.pth"))
    model = model.to(DEVICE)
    model.autoencoder.load_state_dict(ae_checkpoint)
    model.discriminator.load_state_dict(dc_checkpoint)
    model.eval()
    # load segmentation network
    if RUN_ID.lower().startswith("coco"):
        seg = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained="cocostuff10k", n_classes=182)
        seg = seg.to('cuda')
        phi = deeplab
    elif RUN_ID.lower().startswith("city"):
        seg = PSPNet(layers=101, classes=19, zoom_factor=8, pretrained=False)
        seg = seg.to('cuda')
        checkpoint = torch.load(os.path.join(PATH_MODELS, "pspnet_city.pth"))
        seg.load_state_dict(checkpoint['state_dict'], strict=False)
        phi = psp
    seg.eval()
    # compute metrics
    ds, ious, psis = evaluate(model, seg, test_loader)
    # plot samples
    plot_reconstruction(model, test_loader, "test", PATH_FIGS, model.filename)

if __name__ == '__main__':
    main()
