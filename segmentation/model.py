"""
segmentation/model.py — evaluator wrapper
"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

NUM_CLASSES = 21
_MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
_STD  = np.array([0.229,0.224,0.225], dtype=np.float32)
_IMG  = 384


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, d=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, padding=d, dilation=d, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class ASPP(nn.Module):
    def __init__(self, cin, cout=256, rates=(1,6,12,18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(cin, cout, 1 if r==1 else 3, padding=0 if r==1 else r, dilation=r, bias=False),
                          nn.BatchNorm2d(cout), nn.ReLU(inplace=True)) for r in rates])
        self.gp = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1, bias=False),
                                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.proj = nn.Sequential(nn.Conv2d(cout*(len(rates)+1), cout, 1, bias=False),
                                  nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Dropout2d(0.1))
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        gp = F.interpolate(self.gp(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(torch.cat(feats+[gp], dim=1))

class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.reduce = nn.Conv2d(cin, cout, 1)
        self.skip   = nn.Conv2d(cskip, cout, 1)
        self.conv = nn.Sequential(ConvBlock(cout*2, cout), ConvBlock(cout, cout))
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([self.reduce(x), self.skip(skip)], dim=1))

class SegNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        w = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        bb = resnet50(weights=w)
        self.stem=nn.Sequential(bb.conv1, bb.bn1, bb.relu); self.pool=bb.maxpool
        self.layer1=bb.layer1; self.layer2=bb.layer2; self.layer3=bb.layer3; self.layer4=bb.layer4
        self.aspp = ASPP(2048, 256)
        self.up4 = UpBlock(256, 1024, 256); self.up3 = UpBlock(256, 512, 128)
        self.up2 = UpBlock(128, 256, 64);   self.up1 = UpBlock(64, 64, 64)
        self.final = nn.Sequential(ConvBlock(64, 64), nn.Conv2d(64, num_classes, 1))
        self.aux = nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
                                 nn.Conv2d(256, num_classes, 1))
    def forward(self, x):
        H,W = x.shape[-2:]
        s1 = self.stem(x); p = self.pool(s1)
        c2 = self.layer1(p); c3 = self.layer2(c2); c4 = self.layer3(c3); c5 = self.layer4(c4)
        f = self.aspp(c5)
        d4 = self.up4(f, c4); d3 = self.up3(d4, c3); d2 = self.up2(d3, c2); d1 = self.up1(d2, s1)
        return F.interpolate(self.final(d1), size=(H,W), mode="bilinear", align_corners=False)


class SegmentationModel:
    def __init__(self, weights_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegNet(pretrained=False)
        weights_dir = str(weights_dir)
        ck = os.path.join(weights_dir, "segmenter.pth")
        if not os.path.isfile(ck):
            pth = sorted(f for f in os.listdir(weights_dir) if f.endswith(".pth"))
            if not pth: raise FileNotFoundError(f"No .pth in {weights_dir}")
            ck = os.path.join(weights_dir, pth[0])
        state = torch.load(ck, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @staticmethod
    def _prep(image, size=_IMG):
        if image.dtype != np.uint8: image = image.astype(np.uint8)
        if image.ndim == 2: image = np.stack([image]*3, -1)
        if image.shape[-1] == 4: image = image[..., :3]
        H, W = image.shape[:2]
        s = size / max(H, W); nh, nw = int(round(H*s)), int(round(W*s))
        pil = Image.fromarray(image).resize((nw, nh), Image.BILINEAR)
        canv = Image.new("RGB", (size, size), (0,0,0))
        px, py = (size-nw)//2, (size-nh)//2
        canv.paste(pil, (px, py))
        a = (np.asarray(canv, dtype=np.float32)/255.0 - _MEAN) / _STD
        return torch.from_numpy(a.transpose(2,0,1)).float().unsqueeze(0), (H,W,px,py,nw,nh)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2: image = np.stack([image]*3, -1)
        H, W = image.shape[:2]; probs = None
        for size in (_IMG, int(_IMG*1.25)):
            for flip in (False, True):
                x, info = self._prep(image, size)
                _,_,px,py,nw,nh = info; x = x.to(self.device)
                if flip: x = torch.flip(x, [3])
                p = F.softmax(self.model(x), dim=1)
                if flip: p = torch.flip(p, [3])
                p = p[:,:,py:py+nh,px:px+nw]
                p = F.interpolate(p, size=(H,W), mode="bilinear", align_corners=False)
                probs = p if probs is None else probs+p
        return probs.argmax(1)[0].cpu().numpy().astype(np.uint8)
