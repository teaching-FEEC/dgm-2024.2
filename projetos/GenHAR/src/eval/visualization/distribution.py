import numpy as np
import matplotlib as plt
def visualize_distribution(ori_data, gen_data, result_path, save_file_name):
    sample_num = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:sample_num]

    ori_data = ori_data[idx]
    gen_data = gen_data[idx]

    prep_data = np.mean(ori_data, axis=1)
    prep_data_hat = np.mean(gen_data, axis=1)

    fig, ax = plt.subplots(1,1,figsize = (2,2))
    sns.kdeplot(prep_data.flatten(), color='C0', linewidth=2, label='Original', ax=ax)

    # Plotting KDE for generated data on the same axes
    sns.kdeplot(prep_data_hat.flatten(), color='C1', linewidth=2, linestyle='--', label='Generated', ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(0,1)
    for pos in ['top','right']:
        ax.spines[pos].set_visible(False)
    save_path = os.path.join(result_path, 'distribution_'+save_file_name+'.png')
    make_sure_path_exist(save_path)
    plt.savefig(save_path, dpi=400, bbox_inches='tight')