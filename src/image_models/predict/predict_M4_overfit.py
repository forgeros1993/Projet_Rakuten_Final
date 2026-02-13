import torch
import torch.nn as nn
from torchvision import models
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from image_models.preprocessing import get_preprocessing, load_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# on doit redefinir la tete pour charger les poids
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

class PredictorM4:
    def __init__(self, model_path):
        # M4 = ResNet50 + LegendMLP
        self.model = models.resnet50(weights=None)
        self.model.fc = LegendMLP(2048, 27)
        
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE).eval()
        # M4 utilise la transfo standard ImageNet (224px)
        self.transform = get_preprocessing('M3') # Reuse la transfo 224px

    def predict(self, image_path):
        img = load_image(image_path)
        if img is None: return None
        tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.model(tensor)
            return int(output.argmax(1).item())