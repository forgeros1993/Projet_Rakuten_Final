import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
import cv2

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_SIZE = 224

# chemins
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

def train_m3():
    print("lancement train m3 efficientnet")
    
    # chargement data
    path_img = os.path.join(BASE_DIR, "data", "raw", "images", "images")
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "X_train_update.csv"))
    y = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "Y_train_CVw08PX.csv"))
    
    if 'prdtypecode' not in y.columns: y = y.rename(columns={y.columns[1]: 'prdtypecode'})
    df = df.merge(y, left_index=True, right_index=True)
    df['path'] = df.apply(lambda r: os.path.join(path_img, f"image_{r['imageid']}_product_{r['productid']}.jpg"), axis=1)
    df = df[df['path'].apply(os.path.exists)]
    
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['prdtypecode'])
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['label'])
    
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_ds = RakutenDataset(train_df, transform=trans)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # load model effnet b0
    model = models.efficientnet_b0(weights="DEFAULT")
    
    # modif tete
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 27)
    model.to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # train rapide
    for ep in range(5):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        print(f"fin epoch {ep+1}")
        
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "M3_IMAGE_Classic_EfficientNetB0.pth"))
    print("modele m3 sauvegarde")

if __name__ == "__main__":
    train_m3()