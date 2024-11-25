import argparse
import gc
import logging
import os
import sys
import time
import platform

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from sgan.data.loader import data_loader
from sgan.losses import gan_g_loss, gan_d_loss, l2_loss
from sgan.losses import displacement_error, final_displacement_error

from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np

torch.backends.cudnn.benchmark = True

# Global variable
global_checkpoint  = {}
processed_models_d = {}
processed_models   = 0

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='zara1', type=str)
parser.add_argument('--delim', default=' ')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=200, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--num_transformer_blocks', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=1024, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=128, type=int)
parser.add_argument('--noise_dim', default=None, type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=5e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=1024, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=5e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=0, type=float)
parser.add_argument('--best_k', default=1, type=int)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=0, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    elif args.use_gpu == 0:
        long_dtype = torch.LongTensor
        float_dtype = torch.FloatTensor
    return long_dtype, float_dtype


def train_model(args): # aka main function
    global global_checkpoint, processed_models

    logger.info(f'Model: {args.model_name}')
    logger.info(f'Training Dataset: {args.dataset_name}')

    # Checking the study type
    study_name = (
        'first_study' if args.current_study == 1 else
        'second_study' if args.current_study == 2 else
        'third_study' if args.current_study == 3 else
        'unknown_study'
    )

    if study_name == 'unknown_study':
        logger.warning('(Warning) Unknown study specified in args.current_study.')
        return -1

    # Adjusting the model name
    # x  - number of cells
    # ed - embedding dim
    # md - mlp dim
    # hd - hidden dim
    model_name = ''
    if args.model == 'LSTM':
        model_name = f'LSTM_ed{args.embedding_dim}_md{args.mlp_dim}_hd{args.encoder_h_dim_d}'
    elif args.model == 'Transformer':
        model_name = f'Transformer_x{args.num_transformer_blocks}_ed{args.embedding_dim}_md{args.mlp_dim}_hd{args.encoder_h_dim_d}_pe{'T' if args.positional_encoding == True else 'F'}_ap{'T' if args.attention_pooling == True else 'F'}'
    else:
        raise ValueError('Unknown model type!')

    # Initializing global and dataset variables
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)
    
    # Check if the model exists and is already processed
    full_model_name = '{}_{}_{}'.format(model_name, args.num_iterations, args.dataset_name)
    if full_model_name in processed_models_d.get(study_name, []):
        logger.info("The model {} has already been processed in {}...\n".format(model_name, study_name))
        return 0

    # Proceeding with the processing
    logger.info('There are {} iterations per epoch'.format(iterations_per_epoch))
    print('')

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

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Here is the generator:')
    logger.info(generator)

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        num_transformer_blocks=args.num_transformer_blocks,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type,
        transformer_cell=False if args.model == 'LSTM' else True,
        attention_pooling=args.attention_pooling if args.model == 'Transformer' else False,
        positional_encoding=args.positional_encoding if args.model == 'Transformer' else False)

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.d_learning_rate
    )

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }

    # Initialize figure and plots
    fig, axs = start_plot()

    start_time = time.time()
    iteration_times = []

    t0 = None
    while t < args.num_iterations:        
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))

        for batch in train_loader:
            iteration_start = time.time()

            for ax in axs:
                ax.clear()
                ax.relim()
                ax.autoscale_view()
                ax.grid()

            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, generator,
                                              discriminator, d_loss_fn,
                                              optimizer_d)
                checkpoint['norm_d'].append(
                    get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                step_type = 'g'
                losses_g = generator_step(args, batch, generator,
                                          discriminator, g_loss_fn,
                                          optimizer_g)
                checkpoint['norm_g'].append(
                    get_total_norm(generator.parameters())
                )
                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('Iteration {}/{} ({:.2f}%)'.format(t, args.num_iterations, (t / args.num_iterations) * 100))

                for k, v in sorted(losses_d.items()):
                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)
                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    args, val_loader, generator, discriminator, d_loss_fn
                )
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    args, train_loader, generator, discriminator,
                    d_loss_fn, limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()

                if (args.checkpoint):
                    checkpoint_path = os.path.join(
                        args.output_dir, '%s_with_model.pt' % args.checkpoint_name
                    )
                    logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                    torch.save(checkpoint, checkpoint_path)
                    logger.info('Done.')

                    # Save a checkpoint with no model weights by making a shallow
                    # copy of the checkpoint excluding some items
                    checkpoint_path = os.path.join(
                        args.output_dir, '%s_no_model.pt' % args.checkpoint_name)
                    logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                    key_blacklist = [
                        'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                        'g_optim_state', 'd_optim_state', 'd_best_state',
                        'd_best_nl_state'
                    ]
                    small_checkpoint = {}
                    for k, v in checkpoint.items():
                        if k not in key_blacklist:
                            small_checkpoint[k] = v
                    torch.save(small_checkpoint, checkpoint_path)
                    logger.info('Done.')

            # Estimating processing time
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)

            elapsed_time = time.time() - start_time
            elapsed_time_formatted = format_time(elapsed_time)

            avg_time_per_iteration = sum(iteration_times) / len(iteration_times)
            remaining_iterations = args.num_iterations - t - 1
            estimated_time_remaining = avg_time_per_iteration * remaining_iterations
            total_estimated_time = elapsed_time + estimated_time_remaining

            estimated_time_remaining_formatted = format_time(estimated_time_remaining)
            total_estimated_time_formatted = format_time(total_estimated_time)

            print('')
            logger.info(f"Iteration {t}/{args.num_iterations} complete.")
            logger.info(f"Current time/iteration: {iteration_time:.2f}s")
            logger.info(f"Average time/iteration: {avg_time_per_iteration:.2f}s.")
            logger.info(f"Elapsed time: {elapsed_time_formatted}.")
            logger.info(f"Estimated remaining time: {estimated_time_remaining_formatted}.")
            logger.info(f"Total estimated time: {total_estimated_time_formatted}.")
            print('')

            # Plotting Losses, ADE and FDE
            #if (t % 10 == 0): # print the figure every 10 steps
            #    dynamic_plot(checkpoint, axs, args)

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            if t >= args.num_iterations:
                dynamic_plot(checkpoint, axs, args)
                processed_models += 1
                break
        
        
    # Turn off dynamically drawn figure
    plt.ioff()

    # Save model and figure if ablation study is ongoing
    if (args.ablation_study):
        models_path = './models/{}'.format(
            'first_study' if args.current_study == 1 else
            'second_study' if args.current_study == 2 else
            'third_study' if args.current_study == 3 else
            'unknown_study'
        )

        # Save final figure
        fig.savefig(f'{models_path}/figure_{model_name}_{t}_{args.dataset_name}.png')
        plt.close()
        
        # Saving model
        models_path = os.path.join(
            models_path, '{}_{}_{}.pt'.format(model_name, t, args.dataset_name)
        )
        logger.info('Saving final model {} trained with {} iteractions...'.format(model_name, t))
        torch.save(checkpoint, models_path)
        logger.info('Model saved successfully!')

        # Save global checkpoint
        global_checkpoint[args.dataset_name][model_name] = {
            'name'            : model_name,
            'type'            : args.model,
            'dataset_name'    : args.dataset_name,
            'iterations'      : t + 1, # to account the last step
            'checkpoint'      : checkpoint
        }
    
        logger.info('Saving global checkpoint...')
        torch.save(global_checkpoint[args.dataset_name], 'models/global_models_checkpoint_variable_{}.pt'.format(args.dataset_name))
        logger.info('Global checkpoint saved successfully!')


def discriminator_step(
    args, batch, generator, discriminator, d_loss_fn, optimizer_d
):
    batch = [tensor for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(
    args, batch, generator, discriminator, g_loss_fn, optimizer_g
):
    batch = [tensor for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return losses


def check_accuracy(
    args, loader, generator, discriminator, d_loss_fn, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


def start_plot():
    fig, axs = plt.subplots(3, 1, figsize=(16, 9))
    plt.ion()
    fig.suptitle('{} ({})'.format(args.model_name, args.dataset_name))
    return fig, axs


def dynamic_plot(checkpoint, axs, args):
    '''checkpoint = {
        'args': args.__dict__,
        'G_losses': defaultdict(list),
        'D_losses': defaultdict(list),
        ...
        'metrics_val': defaultdict(list),
        'metrics_train': defaultdict(list),
        ...
    }'''

    g_total_loss = checkpoint['G_losses']['G_total_loss']
    d_total_loss = checkpoint['D_losses']['D_total_loss']
    ade_values_val   = checkpoint['metrics_val']['ade']
    ade_values_train = checkpoint['metrics_train']['ade']
    fde_values_val   = checkpoint['metrics_val']['fde']
    fde_values_train = checkpoint['metrics_train']['fde']

    # First graph: Generator and Discriminator Losses
    axs[0].plot(np.arange(len(g_total_loss)), g_total_loss, label='Generator Loss', color='black', linestyle='-')  # Solid line
    axs[0].plot(np.arange(len(d_total_loss)), d_total_loss, label='Discriminator Loss', color='black', linestyle='--')  # Dashed line
    axs[0].set_xscale('log')  # Logarithmic x-axis
    axs[0].set_yscale('log')  # Logarithmic y-axis
    axs[0].set_xlim(1, args.num_iterations)  # Avoid 0 on log scale
    axs[0].set_ylim(0.1, 10)  # Avoid 0 on log scale
    axs[0].set_title('')
    axs[0].set_xlabel('')#('Iterations (log scale)')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Second graph: ADE Training and Validation
    axs[1].plot(np.arange(len(ade_values_train)), ade_values_train, label='ADE Training', color='black', linestyle='-')  # Solid line
    axs[1].plot(np.arange(len(ade_values_val)), ade_values_val, label='ADE Validation', color='black', linestyle='--')  # Dashed line
    axs[1].set_xscale('log')  # Logarithmic x-axis
    axs[1].set_yscale('log')  # Logarithmic y-axis
    axs[1].set_xlim(1, args.num_iterations)  # Avoid 0 on log scale
    axs[1].set_ylim(0.1, 7)  # Avoid 0 on log scale
    axs[1].set_title('')
    axs[1].set_xlabel('')#('Iterations (log scale)')
    axs[1].set_ylabel('ADE')
    axs[1].legend()
    axs[1].grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Third graph: FDE Training and Validation
    axs[2].plot(np.arange(len(fde_values_train)), fde_values_train, label='FDE Training', color='black', linestyle='-')  # Solid line
    axs[2].plot(np.arange(len(fde_values_val)), fde_values_val, label='FDE Validation', color='black', linestyle='--')  # Dashed line
    axs[2].set_xscale('log')  # Logarithmic x-axis
    axs[2].set_yscale('log')  # Logarithmic y-axis
    axs[2].set_xlim(1, args.num_iterations)  # Avoid 0 on log scale
    axs[2].set_ylim(0.1, 7)  # Avoid 0 on log scale
    axs[2].set_title('')
    axs[2].set_xlabel('Training Steps (t)')
    axs[2].set_ylabel('FDE')
    axs[2].legend()
    axs[2].grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Redraw the figure                
    plt.draw()
    plt.pause(0.001)


def format_time(seconds):
    """Convert seconds to a string in the format HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def checkpoint_model_checker(args):
    global global_checkpoint, processed_models, processed_models_d
    
    if (not global_checkpoint) and (not processed_models_d):
        print('')
        logger.info('Resuming the global checkpoint and processed files.')

        # Detect all processed models in the specified study path
        models_dir = './models'
        study_dirs = [
            'first_study',
            'second_study',
            'third_study'
        ]

        for study_dir in study_dirs:
            study_path = os.path.join(models_dir, study_dir)

            # Gather all `.pt` files in the study directory
            if os.path.exists(study_path):
                processed_files = [
                    f for f in os.listdir(study_path) 
                    if os.path.isfile(os.path.join(study_path, f)) and f.endswith('.pt')
                ]
                processed_models_d[study_dir] = [
                    os.path.splitext(f)[0] for f in processed_files
                ]

                processed_models += len(processed_files)

                logger.info(f"Detected {len(processed_files)} processed files in {study_dir}.")

                for f in processed_models_d[study_dir]:
                    logger.info(f'  - {f}')
            else:
                logger.error(f"Study path '{study_path}' does not exist.")
                processed_files = []

        logger.info(f"Total processed models: {processed_models}")
        logger.info(f"Processed models organized by study_dir: {processed_models_d}")
        print('')

        # Check if the global checkpoint exists and load it
        # It'll show future warning due to torch.load security issue
        for file_name in os.listdir('./models'):
            if file_name.startswith('global_models_checkpoint_variable_') and file_name.endswith('.pt'):
                try:
                    # extract the dataset name from the file name
                    dataset_name = file_name.replace('global_models_checkpoint_variable_','').replace('.pt','')

                    # file path
                    file_path = os.path.join('./models', file_name)

                    # loading the checkpoint
                    logger.info(f'Loading {dataset_name} global checkpoint from {file_name}')                    
                    tmp_checkpoint = torch.load(file_path)

                    if isinstance(tmp_checkpoint, dict):
                        global_checkpoint[dataset_name] = tmp_checkpoint
                        logger.info(f"Global checkpoint for dataset '{dataset_name}' successfully loaded.")
                    else:
                        logger.warning(f"Global checkpoint in '{file_name}' is not a valid dictionary. The resulting global checkpoint will be unreliable.")
                except Exception as e:
                    logger.error(f"Failed to load the global checkpoint from '{file_name}'. Error: {e}")
            
        logger.info(f'Total global checkpoints loaded: {len(global_checkpoint)}')
        logger.info(f'Total loaded datasets: {list(global_checkpoint.keys())}')

        if len(global_checkpoint) != 0:
            logger.info("Resulting global checkpoint contents:")

            for d, c in global_checkpoint.items():
                print(f"  - {d}:")
                for m, k in c.items():
                    print(f"    - {m}")
                    for content, v in k.items():
                        if content == 'checkpoint' and isinstance(v, dict):
                            print(f'      - {content}:')
                            for k2, v2 in v.items():
                                # Get the size of the variable in bytes
                                size_in_bytes = sys.getsizeof(v2)
                                if size_in_bytes < 30:  # Example threshold: 1024 bytes
                                    print(f'        - {k2}: {v2}')
                                else:
                                    print(f'        - {k2}: [DATA TOO LARGE TO SHOW - {size_in_bytes} bytes]')
                        else:
                            print(f"      - {content}: {v}")
        print('')


def main(dataset, num_epochs, num_epochs_detailed):
    args.dataset_name = dataset

    if args.ablation_study:
        args.num_epochs = num_epochs

        # 1º Ablation Study: varying the hyperparameters
        if args.first_ablation_study:
            logger.info('Starting the first type study with the {} dataset.'.format(args.dataset_name))
            args.current_study = 1

            for model in models:
                # LSTM model
                if model == 'LSTM':
                    # 4 calculations
                    # bl - baseline
                    #                      bl 
                    embedding_dims  = [8 , 16, 32, 64] # exactly equal to the number of LSTM cells
                    mlp_dim         = [32, 64, 64, 128]
                    encoder_h_dim_d = [32, 64, 64, 128]

                    for i in range(0, len(embedding_dims)):
                        args.model_name      = f'S-GAN LSTM (ed{embedding_dims[i]},md{mlp_dim[i]},hd{encoder_h_dim_d[i]})'
                        args.model           = model
                        args.embedding_dim   = embedding_dims[i]
                        args.mlp_dim         = mlp_dim[i]
                        args.encoder_h_dim_d = encoder_h_dim_d[i]
                        
                        train_model(args)
                        
                # TRANSFORMER model
                elif model == 'Transformer':
                    # 6 calculations
                    # blc - baseline comparison
                    #                       blc
                    embedding_dims   = [8 , 16, 16, 16, 32] # 64
                    mlp_dim          = [32, 64, 64, 64, 64] # 128
                    encoder_h_dim_d  = [32, 64, 64, 64, 64] # 128
                    transformer_blks = [1 , 1 , 2 , 4 , 8 ] # 8

                    for i in range(0, len(embedding_dims)):
                        args.model_name             = f'S-GAN Transformers x{transformer_blks[i]} (ed{embedding_dims[i]},md{mlp_dim[i]},hd{encoder_h_dim_d[i]},T,T)'
                        args.model                  = model
                        args.embedding_dim          = embedding_dims[i]
                        args.mlp_dim                = mlp_dim[i]
                        args.encoder_h_dim_d        = encoder_h_dim_d[i]
                        args.num_transformer_blocks = transformer_blks[i]
                        args.attention_pooling      = True
                        args.positional_encoding    = True

                        train_model(args)

                else:
                    raise ValueError('Unknown model type')
            
        # 2º Ablation Study: deactivating the attention pooling & positional encoding on the transformer cell
        if args.second_ablation_study:
            logger.info('Starting the second type study with the {} dataset.'.format(args.dataset_name))
            args.current_study = 2

            embedding_dims      = [16, 16, 16, 16]
            mlp_dim             = [64, 64, 64, 64]
            encoder_h_dim_d     = [64, 64, 64, 64]
            transformer_blks    = [4 , 4 , 4 , 4 ] # discover what is the best quantities according to the 1st ablation study
            positional_encoding = [0 , 0 , 1 , 1 ] # 0 - False, 1 - True
            attention_pooling   = [0 , 1 , 0 , 1 ] # 0 - False, 1 - True

            for i in range(0, len(positional_encoding)):
                args.model_name             = f'S-GAN Transformer x{transformer_blks[i]} (ed{embedding_dims[i]},md{mlp_dim[i]},hd{encoder_h_dim_d[i]},{'T' if positional_encoding[i] == 1 else 'F'},{'T' if attention_pooling[i] == 1 else 'F'})'
                args.model                  = 'Transformer' # setting as default
                args.embedding_dim          = embedding_dims[i]
                args.mlp_dim                = mlp_dim[i]
                args.encoder_h_dim_d        = encoder_h_dim_d[i]
                args.num_transformer_blocks = transformer_blks[i]
                args.positional_encoding    = True if positional_encoding[i] == 1 else False
                args.attention_pooling      = True if attention_pooling[i] == 1 else False

                train_model(args)
        
        # 3º Study: will run both models with the best convergence
        if args.third_study:
            logger.info('Starting the third type study with the {} dataset.'.format(args.dataset_name))
            args.current_study = 3

            args.num_epochs = num_epochs_detailed
            
            for model in models:
                # LSTM model
                if model == 'LSTM':
                    # 4 calculations
                    embedding_dims  = [16] # exactly equal to the number of LSTM cells
                    mlp_dim         = [64]
                    encoder_h_dim_d = [64]

                    for i in range(0, len(embedding_dims)):
                        args.model_name      = f'S-GAN Vanilla LSTM (ed{embedding_dims[i]},md{mlp_dim[i]},hd{encoder_h_dim_d[i]})'
                        args.model           = model
                        args.embedding_dim   = embedding_dims[i]
                        args.mlp_dim         = mlp_dim[i]
                        args.encoder_h_dim_d = encoder_h_dim_d[i]
                        
                        train_model(args)
                        
                # TRANSFORMER model
                elif model == 'Transformer':
                    # 6 calculations
                    embedding_dims   = [16]
                    mlp_dim          = [64]
                    encoder_h_dim_d  = [64]
                    transformer_blks = [4]

                    for i in range(0, len(embedding_dims)):
                        args.model_name             = f'S-GAN Transformers x{transformer_blks[i]} (ed{embedding_dims[i]},md{mlp_dim[i]},hd{encoder_h_dim_d[i]},T,T)'
                        args.model                  = model
                        args.embedding_dim          = embedding_dims[i]
                        args.mlp_dim                = mlp_dim[i]
                        args.encoder_h_dim_d        = encoder_h_dim_d[i]
                        args.num_transformer_blocks = transformer_blks[i]
                        args.attention_pooling      = True
                        args.positional_encoding    = True

                        train_model(args)

                else:
                    raise ValueError('Unknown model type')

    else:
        args.num_epochs = num_epochs
        args.model_name = 'S-GAN {} x{}'.format(args.model, args.num_transformer_blocks)
        train_model(args)


if __name__ == '__main__':
    os.system('cls' if platform.system() == 'Windows' else 'clear')

    # ensure that the models path exist
    os.makedirs('models/', exist_ok=True)
    os.makedirs('models/first_study', exist_ok=True)
    os.makedirs('models/second_study', exist_ok=True)
    os.makedirs('models/third_study', exist_ok=True)

    # obtain the arguments
    logger.info('Setting the arguments...')
    args = parser.parse_args()

    # Standard args
    args.delim                   = 'tab'
    args.d_type                  = 'local'
    args.pred_len                = 8
    args.encoder_h_dim_g         = 32
    args.encoder_h_dim_d         = 64
    args.decoder_h_dim           = 32
    args.embedding_dim           = 16
    args.bottleneck_dim          = 32
    args.mlp_dim                 = 64
    args.num_layers              = 1
    args.num_transformer_blocks  = 8
    args.noise_dim               = (8,) # single-element tuple
    args.noise_type              = 'gaussian'
    args.noise_mix_type          = 'global'
    args.pool_every_timestep     = 0
    args.l2_loss_weight          = 1
    args.batch_norm              = 0
    args.dropout                 = 0
    args.batch_size              = 32
    args.g_learning_rate         = 1e-3
    args.g_steps                 = 1
    args.d_learning_rate         = 1e-3
    args.d_steps                 = 2
    args.checkpoint_every        = 1
    args.print_every             = 1
    args.num_iterations          = 20000
    #args.num_epochs              = 100
    args.pooling_type            = 'pool_net'
    args.clipping_threshold_g    = 1.5
    args.best_k                  = 10
    args.gpu_num                 = 1
    args.checkpoint_name         = 'gan_test'
    args.checkpoint              = False
    args.restore_from_checkpoint = 0

    # added arguments
    args.dataset_name            = 'zara1'  # defined automatically by the study
    args.dataset_study           = {'zara1'}#, 'zara2', 'univ', 'hotel', 'eth'}
    args.ablation_study          = True
    args.first_ablation_study    = True
    args.second_ablation_study   = True
    args.third_study             = True
    args.current_study           = 0      # attributed automatically

    args.model                   = 'Transformer' # {'LSTM', 'Transformer'}, if and only if the ablation_study is False
    args.attention_pooling       = True
    args.positional_encoding     = True
    
    # Sweep args (Ablation Study)
    models                 = ('LSTM', 'Transformer')
    num_epochs             = 20
    num_epochs_detailed    = 120
    
    # Checking the global checkpoint and existing processed models
    checkpoint_model_checker(args)
    
    # Start studies & training the S-GAN
    logger.info('Starting processing...')
    for dataset in args.dataset_study:
        main(dataset, num_epochs, num_epochs_detailed)
