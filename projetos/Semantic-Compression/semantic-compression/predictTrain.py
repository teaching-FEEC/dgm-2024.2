import os

import cv2
import numpy as np
import pyiqa
import torch
from torch import nn
from torch.utils import data
from torchvision.transforms import v2
from tqdm.auto import tqdm

from config import *
from dataset import SemanticData, SemanticResize
from gan import CGAN
from plots import plot_reconstruction
from segmentation import psp, deeplab, PSPNet

def iou(s, s_hat):
    labels = np.unique(s)
    iou = 0.0
    for label in labels:
        thresh1 = np.where(s == label, 1, 0).astype(np.uint8)
        thresh2 = np.where(s_hat == label, 1, 0).astype(np.uint8)
        intersection = cv2.bitwise_and(thresh1, thresh2)
        union = cv2.bitwise_or(thresh1, thresh2)
        label_iou = np.sum(intersection) / np.sum(union)
        iou += label_iou * np.mean(thresh1)
    return iou

def evaluate(autoencoder, phi, seg, dataloader):
    psnrs = []
    dxs = []
    dss = []
    psis = []
    for batch in tqdm(dataloader):
        x_test = batch['x'].to(DEVICE)
        s_test = batch['s'].to(DEVICE)
        with torch.no_grad():
            if C_MODE == 2:
                x_hat = autoencoder(x_test, s_test.float() / 255.)
            else:
                x_hat = autoencoder(x_test)
            x_test = x_test * 0.5 + 0.5
            x_hat = x_hat * 0.5 + 0.5
            _, _, H, W = x_test.shape
            scale = 256 / min(H, W)
            H0 = int(scale * H)
            W0 = int(scale * W)
            x_test = nn.functional.interpolate(x_test, size=(H0, W0), mode="bilinear", align_corners=False)
            x_hat = nn.functional.interpolate(x_hat, size=(H0, W0), mode="bilinear", align_corners=False)
            s_hat = torch.zeros_like(x_hat[..., :1, :, :])
            for i in range(len(x_hat)):
                s0 = s_test[i].permute(1, 2, 0).cpu().numpy()
                x_np = x_hat[i].permute(1, 2, 0).cpu().numpy()
                x_np = (x_np * 255).astype(np.uint8)
                s_np = phi(seg, x_np)
                s_hat[i] = torch.from_numpy(s_np)
            s_hat = nn.functional.interpolate(s_hat, size=(H, W), mode='nearest-exact')
            s_hat = s_hat.to(s_test.dtype)
            psnr, dx, ds, psi = compute_metrics(x_test, s_test, x_hat, s_hat)
            psnrs.append(psnr)
            dxs.append(dx)
            dss.append(ds)
            psis.append(psi)
    return torch.concat(psnrs), torch.concat(dxs), torch.from_numpy(np.concatenate(dss)), torch.concat(psis)[:, 0]

def compute_metrics(x_test, s_test, x_hat, s_hat):
    psnr = pyiqa.create_metric("psnr", device=DEVICE)
    d = pyiqa.create_metric("ms_ssim", device=DEVICE)
    psi = pyiqa.create_metric("musiq", device=DEVICE)
    ious = []
    for i in range(len(s_test)):
        s0 = s_test[i].permute(1, 2, 0).cpu().numpy()
        s = s_hat[i].permute(1, 2, 0).cpu().numpy()
        ious.append(iou(s0, s))
    return psnr(x_test, x_hat), d(x_test, x_hat), ious, psi(x_hat)

def predict(model, test_loader, epoch):
  
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
    psnrs, ds, ious, psis = evaluate(model.autoencoder, phi, seg, test_loader)
    np.savetxt(os.path.join(PATH_MODELS, "test_" + model.filename + "_" + str(epoch) + ".txt"),
               np.array([(psnrs.cpu().numpy()).mean(), (ds.cpu().numpy()).mean(), (ious.cpu().numpy()).mean(), (psis.cpu().numpy()).mean()]))
    print(psnrs.mean(), ds.mean(), ious.mean(), psis.mean())
