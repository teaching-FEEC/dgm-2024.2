# code to train models
from tqdm.auto import tqdm
from ..gan import GenerativeCompressionGAN
from keras.utils import image_dataset_from_directory
from keras import optimizers, losses


def train(gan, dataloader_train, dataloader_val, epochs, save_every):
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
            gan.save_weights(f'../../models/gan_{epoch}.weights.h5')
        pbar.set_description(f'train: ({tdl:.2f}; {tgl:.2f}); val: ({vdl:.2f}; {vgl:.2f})')
    return d_losses_train, g_losses_train, d_losses_val, g_losses_val


def main():
    path_train = '../../data/train'
    path_val = '../../data/test'
    H, W = 128, 256
    width = 32
    depth = 3
    batch_size = 50
    d_lr = 0.001
    g_lr = 0.001
    epochs = 50
    save_every = 50

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

    train(gan, train_set, val_set, epochs, save_every)

if __name__ == '__main__':
    main()
