#!/usr/bin/env python3
"""
CenterNet / Gaussian Splatting Detection Training — MBP dataset
This script reads YOLO-formatted bounding boxes, converts them to 
centroid heatmaps, and trains a U-Net to predict point sources.
"""

import os
import glob
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── 1. CONFIGURATION ──────────────────────────────────────────
IMG_DIR    = "/home/kdd1/mtseng/data/Astronomy/data/MBP_full_dataset_v2/images"
LBL_DIR    = "/home/kdd1/mtseng/data/Astronomy/data/MBP_full_dataset_v2/labels"
OUT_DIR    = f"/home/kdd1/mtseng/data/Astronomy/centernet_mbp_{time.strftime('%Y%m%d_%H%M%S')}"

IMG_SIZE   = 640  # Resize images to this dimension (UNet prefers multiples of 16/32)
BATCH_SIZE = 8
EPOCHS     = 100
LR         = 1e-4
SIGMA      = 2.0  # Gaussian radius for the heatmap
SEED       = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

# ── 2. HEATMAP GENERATION ─────────────────────────────────────
def generate_heatmap(image_shape, centroids, sigma=2.0):
    H, W = image_shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    
    for x, y in centroids:
        x, y = int(x), int(y)
        if x < 0 or y < 0 or x >= W or y >= H:
            continue
            
        radius = int(math.ceil(3 * sigma))
        x_min, x_max = max(0, x - radius), min(W, x + radius + 1)
        y_min, y_max = max(0, y - radius), min(H, y + radius + 1)
        
        x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
        
        heatmap[y_min:y_max, x_min:x_max] = np.maximum(heatmap[y_min:y_max, x_min:x_max], gaussian)
        
    return torch.from_numpy(heatmap).unsqueeze(0) # (1, H, W)

# ── 3. DATASET CLASS ──────────────────────────────────────────
class MBPHeatmapDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=640, sigma=2.0):
        self.img_size = img_size
        self.sigma = sigma
        
        # Collect matched pairs
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                      glob.glob(os.path.join(img_dir, "*.JPG")))
        
        self.pairs = []
        for img_path in imgs:
            lbl_path = os.path.join(lbl_dir, Path(img_path).stem + ".txt")
            if os.path.isfile(lbl_path):
                self.pairs.append((img_path, lbl_path))
                
        print(f"Dataset initialized with {len(self.pairs)} image-label pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]
        
        # Load Image (Grayscale for Astronomy)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read {img_path}")
            
        orig_h, orig_w = img.shape
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize to [0, 1]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        
        # Load Labels and extract valid centroids
        centroids = []
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Based on your script: Keep only class 1 (valid MBP)
                if line.startswith("1 "):
                    parts = line.strip().split()
                    # YOLO format is normalized: class cx cy w h
                    cx, cy = float(parts[1]), float(parts[2])
                    
                    # Convert normalized coords to absolute resized coordinates
                    abs_x = cx * self.img_size
                    abs_y = cy * self.img_size
                    centroids.append((abs_x, abs_y))
                    
        # Generate target heatmap
        target_heatmap = generate_heatmap((self.img_size, self.img_size), centroids, self.sigma)
        
        return img_tensor, target_heatmap

# ── 4. MODEL ARCHITECTURE (Simple U-Net) ──────────────────────
class SimpleUNetCenterNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(64, 128)
        
        # Decoder 2 (Upsample 160 -> 320)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64) # 128 because 64(upconv) + 64(enc2 skip connection)
        
        # Decoder 1 (Upsample 320 -> 640)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32) # 64 because 32(upconv) + 32(enc1 skip connection)
        
        # Final Output
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder Pass
        e1 = self.enc1(x)                  # Shape: (B, 32, 640, 640)
        e2 = self.enc2(self.pool1(e1))     # Shape: (B, 64, 320, 320)
        
        # Bottleneck
        b = self.bottleneck(self.pool2(e2)) # Shape: (B, 128, 160, 160)
        
        # Decoder Pass 2
        d2 = self.upconv2(b)               # Shape: (B, 64, 320, 320)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=(e2.shape[2], e2.shape[3]), mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)    # Concatenate skip connection
        d2 = self.dec2(d2)                 # Shape: (B, 64, 320, 320)
        
        # Decoder Pass 1
        d1 = self.upconv1(d2)              # Shape: (B, 32, 640, 640)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=(e1.shape[2], e1.shape[3]), mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)    # Concatenate skip connection
        d1 = self.dec1(d1)                 # Shape: (B, 32, 640, 640)
        
        # Final Prediction
        out = self.final_conv(d1)          # Shape: (B, 1, 640, 640)
        return torch.sigmoid(out)          # Bounds probabilities between 0 and 1

# ── 5. LOSS FUNCTION ──────────────────────────────────────────
class HeatmapFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
        
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_weights = torch.pow(1 - target, self.beta)
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.sum()
        if num_pos == 0:
            return -neg_loss.sum()
        else:
            return -(pos_loss.sum() + neg_loss.sum()) / num_pos

# ── 6. TRAINING LOOP ──────────────────────────────────────────
def train():
    print(f"Output directory: {OUT_DIR}")
    
    # Init Data
    dataset = MBPHeatmapDataset(IMG_DIR, LBL_DIR, img_size=IMG_SIZE, sigma=SIGMA)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # Init Model
    model = SimpleUNetCenterNet(in_channels=1, out_channels=1).to(DEVICE)
    criterion = HeatmapFocalLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Track Best Loss
    best_loss = float('inf')
    
    print(f"Starting training on {DEVICE} for {EPOCHS} epochs...\n")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward
            preds = model(images)
            loss = criterion(preds, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print batch progress
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / len(dataloader)
        print(f"==> Epoch {epoch} Average Loss: {avg_loss:.4f}\n")
        
        # Save last model
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "last_centernet.pt"))
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_centernet.pt"))
            print(f"*** New best model saved with loss {best_loss:.4f} ***\n")

if __name__ == "__main__":
    train()