# models/utils.py

import torch
from torch_geometric.data import Data
from PIL import Image
from torchvision import transforms

# -------------------------
# Image preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

def load_image(img):
    """
    img: either a path (str/Path) or a PIL.Image
    returns: torch.Tensor
    """
    if not isinstance(img, Image.Image):
        img = Image.open(img).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img)


# -------------------------
# Build single-node graph from CNN feature
# -------------------------
def build_graph(node_features, edges=None):
    """
    node_features: [1, feature_dim] tensor
    edges: list of (src, dst), default empty
    returns: torch_geometric.data.Data object
    """
    x = node_features
    if edges is None:
        edge_index = torch.zeros((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)
