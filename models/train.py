# models/train.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from cnn_feature_extractor import CNNFeatureExtractor
from gnn_model import GNNClassifier
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import json

# -------------------------
# Parameters
# -------------------------
DATA_DIR = "data/Dataset_3_classes"  # Updated 4-class dataset
CSV_FILE = os.path.join(DATA_DIR, "annotations_3.csv")
BATCH_SIZE = 8
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_MAP_FILE = os.path.join(DATA_DIR, "label_map.json")

# -------------------------
# Dataset
# -------------------------
class PetDataset(Dataset):
    def __init__(self, csv_file, split="train"):
        
        self.data = pd.read_csv(csv_file)
        selected_classes = ["healthy", "dermatitis", "demodicosis"]
        self.data = self.data[self.data['label'].isin(selected_classes)]
        self.data = self.data[self.data['split'] == split].reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        labels = sorted(self.data['label'].unique())
        self.label_map = {label: i for i, label in enumerate(labels)}

        # Save label map
        with open(LABEL_MAP_FILE, "w") as f:
            json.dump(self.label_map, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rel_path = self.data.loc[idx, "filename"]
        img_path = os.path.join(DATA_DIR, rel_path)
        label_name = self.data.loc[idx, "label"]
        label = self.label_map[label_name]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

# -------------------------
# Build graph
# -------------------------
def build_graph(node_features, edges=None):
    if edges is None or len(edges) == 0:
        edges = [(0, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=node_features, edge_index=edge_index)

# -------------------------
# Training
# -------------------------
def train(csv_file, split="train", epochs=EPOCHS, batch_size=BATCH_SIZE):
    dataset = PetDataset(csv_file, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(dataset.label_map)
    cnn = CNNFeatureExtractor(output_dim=128).to(DEVICE)
    gnn = GNNClassifier(input_dim=128, hidden_dim=64, num_classes=num_classes).to(DEVICE)

    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(gnn.parameters()), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            features = cnn(imgs)  # CNN features

            graph_list = [build_graph(f.unsqueeze(0)) for f in features]
            g_batch = Batch.from_data_list(graph_list).to(DEVICE)

            out = gnn(g_batch.x, g_batch.edge_index, g_batch.batch)

            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(cnn.state_dict(), "models/cnn.pth")
    torch.save(gnn.state_dict(), "models/gnn.pth")
    print("✅ Models saved!")

if __name__ == "__main__":
    train(CSV_FILE, split="train", epochs=EPOCHS, batch_size=BATCH_SIZE)
