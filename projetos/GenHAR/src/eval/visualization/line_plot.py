import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_ts_lineplot(
    ts: np.ndarray,
    ys: np.ndarray = None,
    num: int = 5,
    unite_features: bool = True,
    legend_fontsize: int = 12,
    tick_size: int = 10
) -> None:
    """
    Visualizes time series data using line plots.

    This function generates line plots to visualize the time series data. It randomly selects a specified number of samples
    from the input tensor `ts` and plots each sample as a line plot. If `ys` is provided, it can ser either a 1D or 2D tensor
    representing the target variable(s), and the function will optionally overlay it on the line plot.

    :param ts: Input time series data tensor.
    :type ts: np.ndarray
    :param ys: Optional target variable(s) tensor, defaults to None.
    :type ys: np.ndarray, optional
    :param num: Number of samples to visualize, defaults to 5.
    :type num: int, optional
    :param unite_features: Whether to plot all features together or separately, defaults to True.
    :type unite_features: bool, optional
    :param legend_fontsize: Font size to use.
    :type legend_fontsize: int, optional
    :param tick_size: Font size for y-axis ticks.
    :type tick_size: int, optional
    """
    if ys.ndim > 1:
        ys = np.argmax(ys, axis=1)
    assert len(ts.shape) == 3, "O tensor 'ts' deve ter três dimensões (samples, timesteps, features)."

    fig, axs = plt.subplots(num, 1, figsize=(14, 10))
    if num == 1:
        axs = [axs]

    ids = np.random.choice(ts.shape[0], size=num, replace=False)
    for i, sample_id in enumerate(ids):
        if not unite_features:
            feature_id = np.random.randint(ts.shape[2])
            sns.lineplot(
                x=range(ts.shape[1]),
                y=ts[sample_id, :, feature_id],
                ax=axs[i],
                label=rf"Feature #{feature_id}",
            )
        else:
            for feat_id in range(ts.shape[2]):
                sns.lineplot(
                    x=range(ts.shape[1]),
                    y=ts[sample_id, :, feat_id],
                    ax=axs[i],
                    label=f"Feature #{feat_id}",
                )
        if ys is not None:
            axs[i].tick_params(labelsize=tick_size, which="both")
            if len(ys.shape) == 1:
                axs[i].set_title(f"Sample {sample_id} - Label: {ys[sample_id]}", fontsize=legend_fontsize)
            elif len(ys.shape) == 2:
                ax2 = axs[i].twinx()
                sns.lineplot(
                    x=range(ts.shape[1]),
                    y=ys[sample_id],
                    ax=ax2,
                    color="g",
                    label="Condition",
                )
                ax2.tick_params(labelsize=tick_size)
                if i == 0:
                    leg = ax2.legend(fontsize=legend_fontsize, loc='upper right')
                    # Atualização aqui para evitar AttributeError
                    for legobj in leg.get_lines():
                        legobj.set_linewidth(2.0)
                else:
                    ax2.get_legend().remove()
            else:
                raise ValueError("O tensor 'ys' contém muitas dimensões.")
        if i == 0:
            leg = axs[i].legend(fontsize=legend_fontsize, loc='upper left')
            # Atualização aqui para evitar AttributeError
            for legobj in leg.get_lines():
                legobj.set_linewidth(2.0)
        else:
            axs[i].get_legend().remove()
        if i != len(ids) - 1:
            axs[i].set_xticks([])

    plt.tight_layout()
    plt.show()
