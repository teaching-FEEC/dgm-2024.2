# pylint: disable=C0103,E0401,C0411
"""Functions to control training and testing CycleGAN models."""
import time
import gc
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torchvision import transforms
from tqdm import tqdm
import wandb
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / 'src'))
from utils.utils import get_gpu_memory_usage, get_current_commit, remove_all_files, save_dict_as_json, load_json_to_dict
from utils.data_loader import get_img_dataloader
from utils.data_transform import ImageTools
from models.cyclegan import CycleGAN
from models.losses import LossValues, LossLists
from metrics.fid import FID
from metrics.lpips import LPIPS

def save_losses(loss: LossLists, filename='losses.txt'):
    """
    Saves the generator and discriminator losses to a text file.

    Saves a text file containing the losses for the generator and discriminators
    (A and B) over the training epochs.

    Args:
    - loss (LossLists): An instance of LossLists containing lists of losses.
    - filename (str): The file path where the losses will be saved. Defaults to 'losses.txt'.
    """
    df = loss.to_dataframe()
    df = df.rename_axis('Epoch')
    df.to_csv(filename, index=True)


def train_one_epoch(epoch, model, train_A, train_B, device, n_samples=None, plp_step=0):
    """
    Trains the CycleGAN model for a single epoch and returns the generator and discriminator losses.

    Args:
    - epoch (int): The current epoch number.
    - model (CycleGAN): The CycleGAN model instance.
    - train_A (DataLoader): DataLoader for domain A training images.
    - train_B (DataLoader): DataLoader for domain B training images.
    - device (torch.device): The device on which the model and data are
    loaded (e.g., 'cuda' or 'cpu').
    - n_samples (int): Number of samples to train on per batch.
    If None, train on all samples. Default is None.
    - plp_step: Steps between Path Length Penalty calculations. Used to adjust
    PLP loss value. Default is 0.

    Returns:
    - loss_G (float): The total loss of the generator for this epoch.
    - loss_D_A (float): The total loss of discriminator A for this epoch.
    - loss_D_B (float): The total loss of discriminator B for this epoch.

    During training:
    - It iterates through the batches of both domains (A and B) and performs
    optimization on the generators and discriminators.
    - Progress is tracked with a `tqdm` progress bar that shows current generator
    and discriminator losses.
    """
    time_start = time.time()
    model.train()
    progress_bar = tqdm(zip(train_A, train_B), desc=f'Epoch {epoch:03d}',
                        leave=False, disable=False)

    losses_ = LossValues(len(train_A), len(train_B), plp_step)
    for batch_A, batch_B in progress_bar:
        progress_bar.set_description(f'Epoch {epoch:03d}')

        if n_samples is not None:
            batch_A = batch_A[:n_samples]
            batch_B = batch_B[:n_samples]

        real_A = batch_A.to(device)
        real_B = batch_B.to(device)

        loss = model.optimize(real_A, real_B)

        losses_.add(loss)

        progress_bar.set_postfix({
            'G_loss': f'{loss.loss_G.item():.4f}',
            'D_A_loss': f'{loss.loss_D_A.item():.4f}',
            'D_B_loss': f'{loss.loss_D_B.item():.4f}',
            'GPU': f'{get_gpu_memory_usage("",True)}',
        })

        torch.cuda.empty_cache()
        gc.collect()

    progress_bar.close()
    losses_.normalize()
    model.update_lr()

    print(f'Epoch {epoch:03d}: {str(losses_)}, Time={time.time() - time_start:.2f} s')
    return losses_


def evaluate(epoch, model, test_A, test_B, device, n_samples=None, plp_step=0, amp=False):
    """
    Evaluates the CycleGAN model and returns the generator and discriminator losses.

    Args:
    - epoch (int): The current epoch number.
    - model (CycleGAN): The CycleGAN model instance.
    - test_A (DataLoader): DataLoader for domain A testing images.
    - test_B (DataLoader): DataLoader for domain B testing images.
    - device (torch.device): The device on which the model and data are
    loaded (e.g., 'cuda' or 'cpu').
    - n_samples (int): Number of samples to train on per batch.
    If None, train on all samples. Default is None.
    - plp_step: Steps between Path Length Penalty calculations. Used to adjust
    PLP loss value. Default is 0.
    - amp (bool): Whether to use Automatic Mixed Precision (AMP) for training. Default is False.

    Returns:
    - loss_G (float): The total loss of the generator.
    - loss_D_A (float): The total loss of discriminator A.
    - loss_D_B (float): The total loss of discriminator B.
    """
    time_start = time.time()
    model.eval()
    progress_bar = tqdm(zip(test_A, test_B), desc=f'Epoch {epoch:03d}',
                        leave=False, disable=False)

    losses_ = LossValues(len(test_A), len(test_B), plp_step)
    for batch_A, batch_B in progress_bar:
        progress_bar.set_description(f'Epoch {epoch:03d}')

        if n_samples is not None:
            batch_A = batch_A[:n_samples]
            batch_B = batch_B[:n_samples]

        real_A = batch_A.to(device)
        real_B = batch_B.to(device)

        with torch.no_grad():
            loss = model.compute_loss(real_A, real_B, training=False, amp=amp)

        losses_.add(loss)

        progress_bar.set_postfix({
            'G_loss': f'{loss.loss_G.item():.4f}',
            'D_A_loss': f'{loss.loss_D_A.item():.4f}',
            'D_B_loss': f'{loss.loss_D_B.item():.4f}',
            'GPU': f'{get_gpu_memory_usage("",True)}',
        })

        torch.cuda.empty_cache()
        gc.collect()

    progress_bar.close()
    losses_.normalize()

    print(f'     Test: {str(losses_)}, Time={time.time() - time_start:.2f} s')
    return losses_


def init_cyclegan_train(params):
    """Initialize CycleGAN training"""

    def _init_cuda(params):
        params['use_cuda'] = torch.cuda.is_available() and params['use_cuda']
        params['device'] = torch.device("cuda" if params['use_cuda'] else "cpu")
        print(f'Using device: "{params["device"]}"')

        if params['use_cuda']:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            params['print_memory'] = False
        return params

    def _init_new_cycle_gan(params):
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

    def _get_transformation(params):
        return transforms.Compose([
            transforms.Resize(int(params["img_height"] * 1.12),
                                transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop((params["img_height"],
                                    params["img_width"])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _init_dataloaders(params, transformation):
        train_A_csv = params['data_folder'] / f'input_A_train{params["csv_type"]}.csv'
        test_A_csv = params['data_folder'] / f'input_A_test{params["csv_type"]}.csv'
        train_B_csv = params['data_folder'] / f'input_B_train{params["csv_type"]}.csv'
        test_B_csv = params['data_folder'] / f'input_B_test{params["csv_type"]}.csv'

        batch_size = params["batch_size"]
        train_A = get_img_dataloader(csv_file=train_A_csv,
                                    batch_size=batch_size,
                                    transformation=transformation)
        test_A = get_img_dataloader(csv_file=test_A_csv,
                                    batch_size=batch_size,
                                    transformation=transformation)
        train_B = get_img_dataloader(csv_file=train_B_csv,
                                    batch_size=batch_size,
                                    transformation=transformation)
        test_B = get_img_dataloader(csv_file=test_B_csv,
                                    batch_size=batch_size,
                                    transformation=transformation)

        n_train = min(len(train_A.dataset), len(train_B.dataset))
        train_A.dataset.set_len(n_train)
        train_B.dataset.set_len(n_train)
        print(f"Number of training samples: {n_train}")

        n_test = min(len(test_A.dataset), len(test_B.dataset))
        test_A.dataset.set_len(n_test)
        test_B.dataset.set_len(n_test)
        print(f"Number of test samples: {n_test}")

        return train_A, test_A, train_B, test_B


    if params['parameters_path'] is not None:
        params_ = load_json_to_dict(params['parameters_path'])
        for k,v in params_.items():
            if k not in params:
                params[k] = v

    params['out_folder'] = Path(params['out_folder'])
    params['data_folder'] = Path(params['data_folder'])

    params = _init_cuda(params)

    params['out_folder'].mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {params['out_folder']}")

    commit_hash, commit_msg = get_current_commit()
    params['commit_hash'] = commit_hash
    params['commit_msg'] = commit_msg

    model = _init_new_cycle_gan(params)
    if params['restart_path'] is not None:
        params['restart_epoch'] = model.load_model(params['restart_path'])
        print(f"Restarting from {Path(params['restart_path']).name} at epoch {params['restart_epoch']}")
    else:
        params['restart_epoch'] = -1

    if params['print_memory']:
        print(get_gpu_memory_usage("Initital memory usage", short_msg=True))

    transformation = _get_transformation(params)
    train_A, test_A, train_B, test_B = _init_dataloaders(params, transformation)

    fid = FID(dims=2048, cuda=params['use_cuda'], batch_size=128)
    lpips = LPIPS(cuda=params['use_cuda'], batch_size=128)

    return model, (train_A, test_A, train_B, test_B), (fid, lpips)


def train_cyclegan(model, data_loaders, params, metrics):
    """Wrapper function to train the CycleGAN model."""

    if params['run_wandb']:
        wandb.init(
            project="cyclegan",
            name=params['experiment_name'],
            notes=params['experiment_description'],
            config=params)

        wandb.watch(model.gen_AtoB, log_freq=100)
        wandb.watch(model.gen_BtoA, log_freq=100)
        wandb.watch(model.dis_A, log_freq=100)
        wandb.watch(model.dis_B, log_freq=100)

    if params['restart_path'] is None:
        remove_all_files(params['out_folder'])
    save_dict_as_json(params, params['out_folder'] / 'hyperparameters.json')

    train_A, test_A, train_B, test_B = data_loaders
    losses_list = LossLists()

    for epoch in range(params['restart_epoch']+1, params['num_epochs']):
        losses_ = train_one_epoch(
            epoch=epoch,
            model=model,
            train_A=train_A,
            train_B=train_B,
            device=params['device'],
            n_samples=params['n_samples'],
            plp_step=params['plp_step'],
        )

        losses_test_ = evaluate(
            epoch=epoch,
            model=model,
            test_A=test_A,
            test_B=test_B,
            device=params['device'],
            n_samples=params['n_samples'],
            plp_step=params['plp_step'],
            amp=params['amp'],
        )

        # Calculate FID and LPIPS metrics
        fid, lpips = metrics

        with torch.no_grad():
            real_A = next(iter(test_A)).to(params['device'])
            real_B = next(iter(test_B)).to(params['device'])
            fake_A, fake_B = model.generate_samples(real_A, real_B)

            # Calculate FID and LPIPS for A → B
            fid_score_AtoB = fid.get(real_B, fake_B)
            lpips_score_AtoB = lpips.get(real_B, fake_B)

            # Calculate FID and LPIPS for B → A
            fid_score_BtoA = fid.get(real_A, fake_A)
            lpips_score_BtoA = lpips.get(real_A, fake_A)

        print(f'Epoch {epoch:03d} - FID A→B: {fid_score_AtoB:0.4f}, LPIPS A→B: {lpips_score_AtoB.mean():0.4f}')
        print(f'Epoch {epoch:03d} - FID B→A: {fid_score_BtoA:0.4f}, LPIPS B→A: {lpips_score_BtoA.mean():0.4f}')

        losses_list.append(losses_)
        losses_list.append(losses_test_, test=True)

        save_losses(losses_list, filename=params['out_folder'] / 'losses.txt')
        save_checkpoint(model, params, epoch)
        sample_A_path, sample_B_path = save_samples(model, params, test_A, test_B, epoch)

        if params['run_wandb']:
            wandb.log({
                'G_loss/Total/train': losses_.loss_G,
                'G_loss/Adv/train': losses_.loss_G_ad,
                'G_loss/Cycle/train': losses_.loss_G_cycle,
                'G_loss/ID/train': losses_.loss_G_id,
                'G_loss/PLP/train': losses_.loss_G_plp,
                'D_loss/Disc_A/train': losses_.loss_D_A,
                'D_loss/Disc_B/train': losses_.loss_D_B,

                'G_loss/Total/test': losses_test_.loss_G,
                'G_loss/Adv/test': losses_test_.loss_G_ad,
                'G_loss/Cycle/test': losses_test_.loss_G_cycle,
                'G_loss/ID/test': losses_test_.loss_G_id,
                'G_loss/PLP/test': losses_test_.loss_G_plp,
                'D_loss/Disc_A/test': losses_test_.loss_D_A,
                'D_loss/Disc_B/test': losses_test_.loss_D_B,

                'FID_AtoB': fid_score_AtoB,
                'LPIPS_AtoB': lpips_score_AtoB.mean(),
                'FID_BtoA': fid_score_BtoA,
                'LPIPS_BtoA': lpips_score_BtoA.mean(),

                "Samples/Imgs_A": wandb.Image(str(sample_A_path)),
                "Samples/Imgs_B": wandb.Image(str(sample_B_path)),
            })

    if (params['num_epochs']-1) % params['checkpoint_interval'] != 0:
        save_checkpoint(model, params, epoch, force=True)

    return model

def save_samples(model, params, test_A, test_B, epoch):
    """Saves samples of the CycleGAN model."""
    real_A = next(iter(test_A))
    real_B = next(iter(test_B))

    n_images = 4
    if params['use_cuda']:
        real_A = real_A.cuda()
        real_B = real_B.cuda()

    imgs_A, imgs_B = model.generate_samples(real_A, real_B, n_images=n_images)

    imgs_A.to('cpu')
    imgs_B.to('cpu')

    ImageTools.show_img(imgs_A, title=f'Epoch {epoch} - A Images',
                    figsize = (20, 16), nrow=n_images,
                    labels=['Real', 'Fake', 'Recovered', 'Identity'])
    sample_A_path = params['out_folder'] / f'imgs_{epoch}_A.png'
    plt.savefig(sample_A_path)
    plt.close()

    ImageTools.show_img(imgs_B, title=f'Epoch {epoch} - B Images',
                    figsize = (20, 16), nrow=n_images,
                    labels=['Real', 'Fake', 'Recovered', 'Identity'])
    sample_B_path = params['out_folder'] / f'imgs_{epoch}_B.png'
    plt.savefig(sample_B_path)
    plt.close()

    return sample_A_path,sample_B_path

def save_checkpoint(model, params, epoch, force=False):
    """Saves a checkpoint of the CycleGAN model."""
    if (epoch % params['checkpoint_interval'] == 0) or force:
        save_path = params['out_folder'] / f'cycle_gan_epoch_{epoch}.pth'
        model.save_model(save_path, epoch)
        if params['run_wandb']:
            wandb.save(str(save_path), base_path=params['out_folder'])
