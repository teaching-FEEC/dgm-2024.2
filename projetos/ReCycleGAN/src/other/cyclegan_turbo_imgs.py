"""Generate day to night images using the CycleGAN-turbo model.

A new Python virtual environment is recommended to run this script.
The path to img2img-turbo source code must be adjusted as needed.
"""

import sys
from pathlib import Path
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

# Path to CycleGAN-turbo source code
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent / 'img2img-turbo/src'))
from cyclegan_turbo import CycleGAN_Turbo  # pylint: disable=all
from my_utils.training_utils import build_transform  # pylint: disable=all


def model_init(model_name, use_fp16=True):
    """Initialize the CycleGAN-turbo model."""
    model_path = None

    # initialize the model
    gan_model = CycleGAN_Turbo(pretrained_name=model_name, pretrained_path=model_path)
    gan_model.eval()
    gan_model.unet.enable_xformers_memory_efficient_attention()
    if use_fp16:
        gan_model.half()

    return gan_model


def translate_image(model, input_image_path, output_dir, image_prep, direction, prompt, use_fp16=True):
    """Apply the CycleGAN-turbo model to translate an image from day to night or vice versa."""
    t_val = build_transform(image_prep)

    input_image = Image.open(input_image_path).convert('RGB')
    # translate the image
    with torch.no_grad():
        input_img = t_val(input_image)
        x_t = transforms.ToTensor()(input_img)
        x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
        if use_fp16:
            x_t = x_t.half()
        output = model(x_t, direction=direction, caption=prompt)

    output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
    output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

    # save the output image
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / Path(input_image_path).name
    output_pil.save(output_path)


if __name__ == "__main__":

    image_prep_ = 'no_resize' #'resize_512x512'
    direction_ = None
    prompt_ = None

    base_folder = Path(__file__).resolve().parent.parent.parent / 'data/external/nexet'

    model_ = model_init("day_to_night")
    output_dir_  = base_folder / "output_A_turbo"
    output_dir_.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(base_folder / "input_A_train.csv")
    for file_name in tqdm(df['file_name']):
        input_image_ = base_folder / "input_A" / file_name
        translate_image(model_, input_image_, output_dir_, image_prep_, direction_, prompt_)
    df = pd.read_csv(base_folder / "input_A_test.csv")
    for file_name in tqdm(df['file_name']):
        input_image_ = base_folder / "input_A" / file_name
        translate_image(model_, input_image_, output_dir_, image_prep_, direction_, prompt_)

    model_ = None
    torch.cuda.empty_cache()

    model_ = model_init("night_to_day")
    output_dir_  = base_folder / "output_B_turbo"
    output_dir_.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(base_folder / "input_B_train.csv")
    for file_name in tqdm(df['file_name']):
        input_image_ = base_folder / "input_B" / file_name
        translate_image(model_, input_image_, output_dir_, image_prep_, direction_, prompt_)
    df = pd.read_csv(base_folder / "input_B_test.csv")
    for file_name in tqdm(df['file_name']):
        input_image_ = base_folder / "input_B" / file_name
        translate_image(model_, input_image_, output_dir_, image_prep_, direction_, prompt_)
