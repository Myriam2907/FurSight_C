# models/evaluate_verbose_3classes.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from cnn_feature_extractor import CNNFeatureExtractor
from gnn_model import GNNClassifier
from torchvision import transforms
from PIL import Image
import os
import json

# -------------------------
# Parameters
# -------------------------
TEST_DIR = r"data/images_test"  # folder with only 3 classes
LABEL_MAP_FILE = r"data/Dataset_3_classes/label_map.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

# -------------------------
# Dataset
# -------------------------
class TestDataset(Dataset):
    def __init__(self, root_dir, label_map):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.samples = []
        for label_name, idx in label_map.items():
            class_folder = os.path.join(root_dir, label_name)
            if not os.path.exists(class_folder):
                continue
            for fname in os.listdir(class_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_folder, fname), idx))
        self.label_map = label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label, path  # return path for reference

# -------------------------
# Build graph
# -------------------------
def build_graph(node_features, edges=None):
    if edges is None or len(edges) == 0:
        edges = [(0, 0)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=node_features, edge_index=edge_index)

# -------------------------
# Evaluation
# -------------------------
def evaluate(test_dir, label_map_file):
    # Load label map
    with open(label_map_file, "r") as f:
        label_map = json.load(f)

    dataset = TestDataset(test_dir, label_map)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(label_map)
    cnn = CNNFeatureExtractor(output_dim=128).to(DEVICE)
    gnn = GNNClassifier(input_dim=128, hidden_dim=64, num_classes=num_classes).to(DEVICE)

    # Load trained weights
    cnn.load_state_dict(torch.load("models/cnn.pth", map_location=DEVICE))
    gnn.load_state_dict(torch.load("models/gnn.pth", map_location=DEVICE))
    cnn.eval()
    gnn.eval()

    correct_per_class = {k: 0 for k in label_map.keys()}
    total_per_class = {k: 0 for k in label_map.keys()}

    print("\nPredictions per image:")
    print("----------------------")

    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            features = cnn(imgs)
            graph_list = [build_graph(f.unsqueeze(0)) for f in features]
            g_batch = Batch.from_data_list(graph_list).to(DEVICE)
            outputs = gnn(g_batch.x, g_batch.edge_index, g_batch.batch)
            preds = outputs.argmax(dim=1)

            for path, label, pred in zip(paths, labels, preds):
                true_class = [k for k, v in label_map.items() if v == label.item()][0]
                pred_class = [k for k, v in label_map.items() if v == pred.item()][0]
                print(f"{os.path.basename(path)} -> True: {true_class}, Predicted: {pred_class}")

                total_per_class[true_class] += 1
                if true_class == pred_class:
                    correct_per_class[true_class] += 1

    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for cls in label_map.keys():
        if total_per_class[cls] > 0:
            acc = correct_per_class[cls] / total_per_class[cls]
            print(f"{cls}: {acc*100:.2f}% ({correct_per_class[cls]}/{total_per_class[cls]})")
        else:
            print(f"{cls}: No samples found")

if __name__ == "__main__":
    evaluate(TEST_DIR, LABEL_MAP_FILE)
