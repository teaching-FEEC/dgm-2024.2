# pylint: disable=C0413,E0401
"""Script to Test a CycleGAN model"""
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# from metrics import FID, LPIPS
# from utils.data_loader import get_img_dataloader
from src.utils import remove_all_files, load_json_to_dict
# from utils import show_img, image_folder_to_tensor
# from src.utils.run import init_cyclegan_train
from src.models.cyclegan import CycleGAN

def init_new_cycle_gan(params):
    """Initializes a new CycleGAN model."""
    return CycleGAN(
        input_nc=params["channels"],
        output_nc=params["channels"],
        device=params["device"],
        n_features=params["n_features"],
        n_residual_blocks=params["n_residual_blocks"],
        n_downsampling=params["n_downsampling"],
        norm_type=params["norm_type"],
        add_skip=params["add_skip"],
        use_replay_buffer=params["use_replay_buffer"],
        replay_buffer_size=params["replay_buffer_size"],
        vanilla_loss=params["vanilla_loss"],
        cycle_loss_weight=params["cycle_loss_weight"],
        id_loss_weight=params["id_loss_weight"],
        plp_loss_weight=params["plp_loss_weight"],
        plp_step=params["plp_step"],
        plp_beta=params["plp_beta"],
        lr=params["lr"],
        beta1=params["beta1"],
        beta2=params["beta2"],
        step_size=params["step_size"],
        gamma=params["gamma"],
        amp=params["amp"],
    )

def translate_image(model, input_image_path, output_dir):
    """Translate an image."""
    input_img = Image.open(input_image_path).convert('RGB')
    with torch.no_grad():
        x_t = transforms.ToTensor()(input_img)
        x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
        output = model(x_t)

    output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
    output_path = Path(output_dir) / Path(input_image_path).name
    output_pil.save(output_path)

if __name__ == '__main__':
    base_folder = Path(__file__).resolve().parent.parent.parent
    parameters = {
        'restart_path': base_folder / 'no_sync/test_model_7/cycle_gan_epoch_14.pth',
        'parameters_path': base_folder / 'no_sync/test_model_7/hyperparameters.json',

        'data_folder': base_folder / 'data/external/nexet',
        'output_name': 'recycle',
        'csv_type': '_filtered',

        'use_cuda': True,
        'batch_size' : 64,
    }

    params_ = load_json_to_dict(parameters['parameters_path'])
    for k,v in params_.items():
        if k not in parameters:
            parameters[k] = v

    model = init_new_cycle_gan(parameters)
    parameters['restart_epoch'] = model.load_model(parameters['restart_path'])
    print(f"Restarting from {Path(parameters['restart_path']).name}")
    print(f"  Epoch: {parameters['restart_epoch']}")

    data_folder = parameters["data_folder"]

    output_dir_  = data_folder / "output_A_recycle"
    output_dir_.mkdir(parents=True, exist_ok=True)
    remove_all_files(output_dir_)
    df = pd.read_csv(data_folder / "input_A_train.csv")
    for file_name in tqdm(df['file_name']):
        input_image_ = data_folder / "input_A" / file_name
        translate_image(model.gen_AtoB, input_image_, output_dir_)
    df = pd.read_csv(data_folder / "input_A_test.csv")
    for file_name in tqdm(df['file_name']):
        input_image_ = data_folder / "input_A" / file_name
        translate_image(model.gen_AtoB, input_image_, output_dir_)

    output_dir_  = data_folder / "output_B_recycle"
    output_dir_.mkdir(parents=True, exist_ok=True)
    remove_all_files(output_dir_)
    df = pd.read_csv(data_folder / "input_B_train.csv")
    for file_name in tqdm(df['file_name']):
        input_image_ = data_folder / "input_B" / file_name
        translate_image(model.gen_BtoA, input_image_, output_dir_)
    df = pd.read_csv(data_folder / "input_B_test.csv")
    for file_name in tqdm(df['file_name']):
        input_image_ = data_folder / "input_B" / file_name
        translate_image(model.gen_BtoA, input_image_, output_dir_)
