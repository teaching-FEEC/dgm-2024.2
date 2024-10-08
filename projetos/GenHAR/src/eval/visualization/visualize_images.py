import tsgm
import matplotlib.pyplot as plt
import numpy as np
def visualize_original_and_reconst_ts(
    original: tsgm.types.Tensor,
    reconst: tsgm.types.Tensor,
    num: int = 5,
    vmin: int = 0,
    vmax: int = 1,
) -> None:
    """
    Visualizes original and reconstructed time series data.

    This function generates side-by-side visualizations of the original and reconstructed time series data.
    It randomly selects a specified number of samples from the input tensors `original` and `reconst` and
    displays them as images using imshow.

    :param original: Original time series data tensor.
    :type original: tsgm.types.Tensor
    :param reconst: Reconstructed time series data tensor.
    :type reconst: tsgm.types.Tensor
    :param num: Number of samples to visualize, defaults to 5.
    :type num: int, optional
    :param vmin: Minimum value for colormap normalization, defaults to 0.
    :type vmin: int, optional
    :param vmax: Maximum value for colormap normalization, defaults to 1.
    :type vmax: int, optional
    """
    
    assert original.shape == reconst.shape

    fig, axs = plt.subplots(num, 2, figsize=(14, 10))

    ids = np.random.choice(original.shape[0], size=num, replace=False)
    for i, sample_id in enumerate(ids):
        axs[i, 0].imshow(original[sample_id].T, aspect="auto", vmin=vmin, vmax=vmax)
        axs[i, 1].imshow(reconst[sample_id].T, aspect="auto", vmin=vmin, vmax=vmax)

def plot_sample_comparison(x_real, y_real, x_gen, y_gen, label=None, n_samples=100, reshape=False):
    if label is not None:
        indices = np.where(y_real == label)[0]
        selected_real = np.random.choice(indices, min(n_samples, len(indices)), replace=False)
        indices_gen = np.where(y_gen == label)[0]
        selected_gen = np.random.choice(indices_gen, min(n_samples, len(indices_gen)), replace=False)
    else:
        selected_real = np.random.choice(len(y_real), min(n_samples, len(y_real)), replace=False)
        selected_gen = np.random.choice(len(y_gen), min(n_samples, len(y_gen)), replace=False)

    real_samples = x_real[selected_real]
    gen_samples = x_gen[selected_gen]

    if reshape:
        real_samples = real_samples.reshape(-1, 60, 6)
        gen_samples = gen_samples.reshape(-1, 60, 6)

    # Plot
    plt.figure(figsize=(12, n_samples * 3))
    
    for i in range(min(n_samples, len(real_samples))):
        plt.subplot(n_samples, 2, 2 * i + 1)
        plt.title(f'Real {label}' if label is not None else 'Real')
        plt.imshow(real_samples[i].reshape(60, 6), aspect='auto')
    
        plt.subplot(n_samples, 2, 2 * i + 2)
        plt.title(f'Generated {label}' if label is not None else 'Generated')
        plt.imshow(gen_samples[i].reshape(60, 6), aspect='auto')

    plt.tight_layout()
    plt.show()

