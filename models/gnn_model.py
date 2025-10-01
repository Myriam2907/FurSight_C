# models/gnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GNNClassifier, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, edge_index, batch):
        # x: node features, edge_index: graph edges, batch: batch assignments
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # pool nodes per graph
        x = self.fc(x)
        return x

# Example usage
if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    # Dummy graph with 4 nodes
    x = torch.randn(4, 128)
    edge_index = torch.tensor([[0,1,2,3],[1,0,3,2]], dtype=torch.long)
    batch = torch.tensor([0,0,1,1])
    model = GNNClassifier(input_dim=128, hidden_dim=64, num_classes=3)
    out = model(x, edge_index, batch)
    print(out.shape)  # should be (2, 3)
