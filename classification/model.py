"""
classification/model.py — evaluator wrapper
"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision import models as tvm

CLASSES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
           "chair","cow","diningtable","dog","horse","motorbike","person",
           "pottedplant","sheep","sofa","train","tvmonitor"]
NUM_CLASSES = 20
MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)
_IMG = 224


class ClfHead(nn.Module):
    def __init__(self, in_f=2048, hidden=512, n=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(in_f, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(hidden, n))
    def forward(self, x): return torch.sigmoid(self.net(x))


class ClfModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        w = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        bb = tvm.resnet50(weights=w)
        self.backbone = nn.Sequential(*list(bb.children())[:-1])
        self.head = ClfHead()
    def forward(self, x): return self.head(self.backbone(x).flatten(1))


class ClassificationModel:
    def __init__(self, weights_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClfModel(pretrained=False)
        weights_dir = str(weights_dir)
        ck = os.path.join(weights_dir, "classifier.pth")
        if not os.path.isfile(ck):
            pth = sorted(f for f in os.listdir(weights_dir) if f.endswith(".pth"))
            if not pth: raise FileNotFoundError(f"No .pth in {weights_dir}")
            ck = os.path.join(weights_dir, pth[0])
        state = torch.load(ck, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.tf = T.Compose([T.Resize((_IMG, _IMG)), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        if image.dtype != np.uint8: image = image.astype(np.uint8)
        if image.ndim == 2: image = np.stack([image]*3, -1)
        if image.shape[-1] == 4: image = image[..., :3]
        pil = Image.fromarray(image).convert("RGB")
        x = self.tf(pil).unsqueeze(0).to(self.device)
        p = self.model(x)[0].cpu().numpy()
        return {CLASSES[i]: float(p[i]) for i in range(NUM_CLASSES)}
