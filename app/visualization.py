# app/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_heatmap(image_tensor, importance_score, save_path=None):
    """
    image_tensor: torch.Tensor [3,H,W]
    importance_score: float or numpy array same size as image
    """
    img = image_tensor.permute(1,2,0).numpy()
    heatmap = np.ones_like(img[:,:,0]) * importance_score  # dummy heatmap
    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    if save_path:
        plt.savefig(save_path)
    plt.show()
