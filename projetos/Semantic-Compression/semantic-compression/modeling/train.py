
import argparse
from tqdm.auto import tqdm
from gan import GenerativeCompressionGAN
from keras.utils import image_dataset_from_directory
from keras import optimizers, losses
import numpy as np


def train(gan, dataloader_train, dataloader_val, epochs, save_every, model_fname):
    d_losses_train = []
    g_losses_train = []
    d_losses_val = []
    g_losses_val = []
    pbar = tqdm(range(1, epochs+1))
    for epoch in pbar:
        tdl, tgl = gan.train_step(dataloader_train)
        vdl, vgl = gan.evaluate(dataloader_val)
        d_losses_train.append(tdl)
        g_losses_train.append(tgl)
        d_losses_val.append(vdl)
        g_losses_val.append(vgl)
        if epoch % save_every == 0:
            gan.save_weights(f'../models/{model_fname}_{epoch}.weights.h5')
        pbar.set_description(f'train: ({tdl:.2f}; {tgl:.2f}); val: ({vdl:.2f}; {vgl:.2f})')
    return d_losses_train, g_losses_train, d_losses_val, g_losses_val


def main():
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

    model_fname = f"{H:04d}{W:04d}{width:03d}{depth:02d}{batch_size:03d}{int(d_lr*1e5):05d}{int(g_lr*1e5):05d}"

    train_set = image_dataset_from_directory(
        path_train, labels=None, batch_size=batch_size, image_size=(H, W), shuffle=True
    )
    val_set = image_dataset_from_directory(
        path_val, labels=None, batch_size=batch_size, image_size=(H, W), shuffle=True
    )
    
    gan = GenerativeCompressionGAN(
        e_filters=width, e_blocks=depth,
        g_filters=width, g_blocks=depth,
        d_filters=width,
        L=5, c_min=-2, c_max=2,
        lambda_d=10
    )
    gan.compile(
        d_optimizer=optimizers.Adam(learning_rate=d_lr),
        g_optimizer=optimizers.Adam(learning_rate=g_lr),
        gan_loss_fn=losses.MeanSquaredError(),
        distortion_loss_fn=losses.MeanSquaredError()
    )

    d_losses_train, g_losses_train, d_losses_val, g_losses_val = train(gan, train_set, val_set, epochs, save_every, model_fname)
    np.savetxt(f"../models/{model_fname}_losses.txt", np.array([d_losses_train, g_losses_train, d_losses_val, g_losses_val]))

if __name__ == '__main__':
    main()
