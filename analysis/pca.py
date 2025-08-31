import torch
import matplotlib

matplotlib.use('Agg')
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using PCA.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save PCA
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'
    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(features)
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)