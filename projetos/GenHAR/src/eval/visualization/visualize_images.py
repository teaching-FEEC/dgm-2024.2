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
