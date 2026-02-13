import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# config
DEVICE = torch.device("cuda")
BASE_DIR = r"C:\Users\amisf\Desktop\datascientest_projet"
OUT_DIR = os.path.join(BASE_DIR, "implementation", "outputs")

# architecture legend (mlp custom)
class LegendMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 512),  nn.BatchNorm1d(512),  nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x): return self.net(x)

def train_m4():
    print("lancement train m4 (overfit phoenix)")
    
    # chargement npy
    try:
        x_all = np.load(os.path.join(OUT_DIR, 'train_features_resnet50_augmented.npy'))
        y_all_raw = np.load(os.path.join(OUT_DIR, 'train_labels_augmented.npy'))
    except:
        print("fichiers npy introuvables"); return

    le = LabelEncoder()
    y_all = le.fit_transform(y_all_raw)
    x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, stratify=y_all)
    
    train_ds = TensorDataset(torch.tensor(x_train).float().to(DEVICE), torch.tensor(y_train).long().to(DEVICE))
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
    
    # mlp training
    mlp = LegendMLP(2048, len(le.classes_)).to(DEVICE)
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for ep in range(20):
        mlp.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = mlp(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        print(f"epoch {ep+1} ok")

    # assemblage final resnet + mlp
    full_model = models.resnet50(weights="IMAGENET1K_V1")
    full_model.fc = mlp
    torch.save(full_model.state_dict(), os.path.join(OUT_DIR, "M4_IMAGE_DeepLearning_OVERFIT_ResNet.pth"))
    print("modele m4 sauvegarde")

if __name__ == "__main__":
    train_m4()