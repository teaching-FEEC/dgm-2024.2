import argparse
from tqdm.auto import tqdm
from gan import GCGAN, AutoEncoder
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.saving import load_model
from keras.callbacks import EarlyStopping
from keras import optimizers, losses
import numpy as np


def pre_train(model, x_train, x_val, batch_size, epochs, model_fname):
    #callback = EarlyStopping(patience=5, start_from_epoch=5)
    model.fit(x=x_train, y=x_train, validation_data=(x_val, x_val), batch_size=batch_size, epochs=epochs)
    model.save_weights(f'../models/ae_{model_fname}.weights.h5')
    return model


def fine_tune(model, dataloader_train, dataloader_val, epochs, discriminate_every, save_every, model_fname):
    gan_losses_train = []
    l2_losses_train = []
    d_losses_train = []
    gan_losses_val = []
    l2_losses_val = []
    d_losses_val = []
    losses_train = []
    losses_val = []
    pbar = tqdm(range(1, epochs+1))
    for epoch in pbar:

        gan_train = 0.
        l2_train = 0.
        d_train = 0.
        loss_train = 0.
        i = 0
        r_batch = 3

        # if epoch % discriminate_every ==0:
        #     r_batch = 1
        # else:
        #     r_batch = 3

        for batch in dataloader_train:
            i+=1
            b_gan, b_l2, b_d, b_loss = model.train_step(batch, discriminating=(i % discriminate_every == 0))
            gan_train += b_gan
            l2_train += b_l2
            d_train += b_d
            loss_train += b_loss
        gan_losses_train.append(gan_train / len(dataloader_train))
        l2_losses_train.append(l2_train / len(dataloader_train))
        d_losses_train.append(d_train / len(dataloader_train))
        losses_train.append(loss_train / len(dataloader_train))

        gan_val = 0.
        l2_val = 0.
        d_val = 0.
        loss_val = 0.
        for batch in dataloader_val:
            b_gan, b_l2, b_d, b_loss = model.test_step(batch)
            gan_val += b_gan
            l2_val += b_l2
            d_val += b_d
            loss_val += b_loss
        gan_losses_val.append(gan_val / len(dataloader_val))
        l2_losses_val.append(l2_val / len(dataloader_val))
        d_losses_val.append(d_val / len(dataloader_val))
        losses_val.append(loss_val / len(dataloader_val))

        pbar.set_description(f'train=({losses_train[-1]:.2f}, {d_losses_train[-1]:.2f}, {l2_losses_train[-1]:.2f}); val=({losses_val[-1]:.2f}, {d_losses_val[-1]:.2f}, {l2_losses_val[-1]:.2f})')

        if epoch % save_every == 0:
            model.save(f'../models/{model_fname}_{epoch}.keras')
            model.save_weights(f'../models/{model_fname}_{epoch}.weights.h5')
        
    return model, gan_losses_train, l2_losses_train, d_losses_train, losses_train, gan_losses_val, l2_losses_val, d_losses_val, losses_val


def main(path_train, path_val, H, W, filters, n_blocks, channels, momentum, batch_size, ae_lr, gan_lr, epochs_pt, epochs_ft, discriminate_every, save_every):
    model_fname = f"{H:04d}{W:04d}{filters:03d}{n_blocks:02d}{channels:01d}{int(momentum*100):02d}{batch_size:03d}{int(ae_lr*1e5):05d}{int(gan_lr*1e5):05d}"

    train_set = image_dataset_from_directory(
        path_train, labels=None, batch_size=batch_size, image_size=(H, W), shuffle=True
    )
    val_set = image_dataset_from_directory(
        path_val, labels=None, batch_size=batch_size, image_size=(H, W), shuffle=True
    )

    x_train = []
    for batch in train_set:
        x_train.append(batch)
        x_train = tf.concat(x_train, axis=0)

    x_val = []
    for batch in val_set:
        x_val.append(batch)
        x_val = tf.concat(x_val, axis=0)

    ae = AutoEncoder(filters, n_blocks, channels, n_convs=4, sigma=1000., levels=5, l_min=-2, l_max=2, momentum=momentum)
    ae.compile(optimizer=optimizers.Adam(ae_lr), loss='mean_squared_error')
    ae = pre_train(ae, x_train/255., x_val/255., batch_size, epochs_pt, model_fname)
    
    
    gan = GCGAN(
        filters, n_blocks, channels, n_convs=4, sigma=1000, levels=5, l_min=-2, l_max=2, lambda_d=10, momentum=momentum,
    )
    gan.compile(
        optimizer_1=optimizers.Adam(learning_rate=ae_lr),
        optimizer_2=optimizers.Adam(learning_rate=gan_lr),
    )
    gan.autoencoder = load_model(f'../models/ae_{model_fname}.keras')

    gan, gan_losses_train, l2_losses_train, gan_losses_val, l2_losses_val = fine_tune(gan, train_set, val_set, epochs_ft, discriminate_every, save_every, model_fname)
    np.savetxt(f"../models/{model_fname}_losses.txt", np.array([gan_losses_train, l2_losses_train, gan_losses_val, l2_losses_val]))
    return gan

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treinamento de Generative Compression GAN")

    parser.add_argument('--path_train', type=str, default='../data/train', help='Caminho para o dataset de treino')
    parser.add_argument('--path_val', type=str, default='../data/test', help='Caminho para o dataset de validação')
    parser.add_argument('--H', type=int, default=256, help='Altura das imagens')
    parser.add_argument('--W', type=int, default=512, help='Largura das imagens')
    parser.add_argument('--width', type=int, default=128, help='Número de filtros de largura do GAN')
    parser.add_argument('--depth', type=int, default=5, help='Número de blocos de profundidade do GAN')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamanho do batch para treinamento')
    parser.add_argument('--d_lr', type=float, default=0.001, help='Taxa de aprendizado do discriminador')
    parser.add_argument('--g_lr', type=float, default=0.001, help='Taxa de aprendizado do gerador')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas de treinamento')
    parser.add_argument('--save_every', type=int, default=100, help='Salvar pesos a cada número de épocas')

    args = parser.parse_args()

    path_train = args.path_train
    path_val = args.path_val
    H = args.H
    W = args.W
    width = args.width
    depth = args.depth
    batch_size = args.batch_size
    d_lr = args.d_lr
    g_lr = args.g_lr
    epochs = args.epochs
    save_every = args.save_every

    main(path_train, path_val, H, W, width, depth, batch_size, d_lr, g_lr, epochs, save_every)
