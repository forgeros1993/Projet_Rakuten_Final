import torch
import torch.nn as nn
import torch.optim as optim
import timm
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2

# config
DEVICE = torch.device("cuda")
BATCH_SIZE = 4
ACCUM_STEPS = 16
IMG_SIZE = 518
BASE_DIR = r"C:\Users\amisf\Desktop\datascientest_projet"
OUT_DIR = os.path.join(BASE_DIR, "implementation", "outputs")

class RakutenDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df['path'].values
        self.labels = df['label'].values
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform: img = self.transform(img)
            return img, torch.tensor(self.labels[idx], dtype=torch.long)
        except:
            return torch.zeros((3, IMG_SIZE, IMG_SIZE)), torch.tensor(self.labels[idx], dtype=torch.long)

def train_m1():
    print("lancement train m1 (dino deep learning)")
    # data setup
    path_img = os.path.join(BASE_DIR, "data", "raw", "images", "images")
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "X_train_update.csv"))
    y = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "Y_train_CVw08PX.csv"))
    if 'prdtypecode' not in y.columns: y = y.rename(columns={y.columns[1]: 'prdtypecode'})
    df = df.merge(y, left_index=True, right_index=True)
    df['path'] = df.apply(lambda r: os.path.join(path_img, f"image_{r['imageid']}_product_{r['productid']}.jpg"), axis=1)
    df = df[df['path'].apply(os.path.exists)]
    
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['prdtypecode'])
    train_df, _ = train_test_split(df, test_size=0.15, stratify=df['label'])
    
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(RakutenDataset(train_df, trans), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # model dino reg4
    model = timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', pretrained=True, num_classes=27)
    model.to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # boucle rapide demo
    for ep in range(1):
        model.train()
        for i, (bx, by) in enumerate(train_loader):
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(bx)
                loss = criterion(out, by) / ACCUM_STEPS
            scaler.scale(loss).backward()
            if (i+1) % ACCUM_STEPS == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "M1_IMAGE_DeepLearning_DINOv3.pth"))

if __name__ == "__main__":
    train_m1()