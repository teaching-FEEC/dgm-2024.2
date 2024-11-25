# python plot_model03_MOD_MPED_1CENA_VP_KDE_CD.py --model_path models/01 --output_filename top10traj54cd.png --num_samples 5000

import argparse
import os
import torch
import numpy as np
from attrdict import AttrDict
import matplotlib.pyplot as plt
import seaborn as sns

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.utils import relative_to_abs, get_dset_path


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Calcula o erro de deslocamento euclidiano.
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode='sum'):
    """
    Calcula o erro euclidiano do deslocamento final.
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate(args, loader, generator, num_samples, output_filename):
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            # Selecionando uma única sequência do batch
            seq_start, seq_end = seq_start_end[0]
            obs_traj = obs_traj[:, seq_start:seq_end, :]
            pred_traj_gt = pred_traj_gt[:, seq_start:seq_end, :]
            num_ped = seq_end - seq_start

            obs_len = obs_traj.shape[0]
            pred_len = pred_traj_gt.shape[0]

            # Gerar uma paleta de cores fixa para cada pedestre
            colors = sns.color_palette("tab10", num_ped)

            # Configurando o gráfico
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(-5, 15)
            ax.set_ylim(-5, 15)
            ax.set_xlabel("Eixo X")
            ax.set_ylabel("Eixo Y")
            ax.set_title("Distribuição de Probabilidade (KDE) por Pedestre")

            # Plotando as trajetórias observadas
            for i in range(num_ped):
                ax.plot(
                    obs_traj[:, i, 0].cpu().numpy(),
                    obs_traj[:, i, 1].cpu().numpy(),
                    color=colors[i],
                    linestyle='-',
                    label=f"Observado ID {i}"
                )

            # Plotando as trajetórias reais
            for i in range(num_ped):
                ax.plot(
                    pred_traj_gt[:, i, 0].cpu().numpy(),
                    pred_traj_gt[:, i, 1].cpu().numpy(),
                    color=colors[i],
                    linestyle='--',
                    label=f"Real ID {i}"
                )

            # Gerar KDE para cada pedestre
            for i in range(num_ped):
                all_trajectories = []
                for _ in range(num_samples):
                    pred_traj_fake_rel = generator(
                        obs_traj, obs_traj_rel[:, seq_start:seq_end, :], seq_start_end[:1]
                    )
                    pred_traj_fake = relative_to_abs(
                        pred_traj_fake_rel, obs_traj[-1]
                    )
                    all_trajectories.append(pred_traj_fake[:, i, :].cpu().numpy())

                # Combinar todas as predições para o pedestre i
                all_trajectories = np.concatenate(all_trajectories, axis=0)

                # Gerar KDE para o pedestre i
                sns.kdeplot(
                    x=all_trajectories[:, 0],
                    y=all_trajectories[:, 1],
                    ax=ax,
                    color=colors[i],
                    fill=True,
                    alpha=0.3,
                    label=f"KDE ID {i}"
                )

            ax.legend(loc="upper left")
            plt.savefig(output_filename, dpi=300)
            plt.show()
            plt.close()
            break  # Processar apenas o primeiro batch


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path, weights_only=False)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader(_args, path)
        evaluate(_args, loader, generator, args.num_samples, args.output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_samples', default=1000, type=int)
    parser.add_argument('--dset_type', default='test', type=str)
    parser.add_argument('--output_filename', type=str, required=True)
    args = parser.parse_args()
    main(args)
