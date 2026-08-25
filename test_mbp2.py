#!/usr/bin/env python3
"""
Tests CenterNet on 50 randomly selected unseen images from results.txt
Generates 2-panel side-by-side plots (Ground Truth vs Prediction).
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ── 1. CONFIGURATION ──────────────────────────────────────────
# Paths to your datasets and model
RESULTS_FILE  = "results.txt"  # Assumes this is in the same directory you run the script from
IMG_DIR       = "/home/kdd1/mtseng/data/Astronomy/data/MBP_full_dataset/images"
LBL_DIR       = "/home/kdd1/mtseng/data/Astronomy/data/MBP_full_dataset/labels"

# Model directory (where weights are loaded from and images are saved to)
MODEL_DIR     = "/home/kdd1/mtseng/data/Astronomy/centernet_mbp_20260401_093004"
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "best_centernet.pt")

IMG_SIZE      = 640
CONF_THRESH   = 0.3  # Minimum heatmap probability to count as an MBP
NUM_TEST_IMGS = 50   # Number of unseen images to test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

# ── 2. MODEL ARCHITECTURE (Must match training exactly) ───────
class SimpleUNetCenterNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(64, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64) 
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32) 
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
        e1 = self.enc1(x)                  
        e2 = self.enc2(self.pool1(e1))     
        b = self.bottleneck(self.pool2(e2)) 
        d2 = self.upconv2(b)               
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=(e2.shape[2], e2.shape[3]), mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)    
        d2 = self.dec2(d2)                 
        d1 = self.upconv1(d2)              
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=(e1.shape[2], e1.shape[3]), mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)    
        d1 = self.dec1(d1)                 
        out = self.final_conv(d1)          
        return torch.sigmoid(out)          

# ── 3. PEAK EXTRACTION (NMS) ──────────────────────────────────
def extract_peaks(heatmap_tensor, threshold=0.3, pool_kernel=3):
    """Extracts (x, y) coordinates from the predicted heatmap."""
    pad = (pool_kernel - 1) // 2
    hmax = F.max_pool2d(heatmap_tensor, pool_kernel, stride=1, padding=pad)
    keep = (hmax == heatmap_tensor).float() * (heatmap_tensor >= threshold).float()
    
    batch_indices, channel_indices, y_coords, x_coords = torch.nonzero(keep, as_tuple=True)
    scores = heatmap_tensor[batch_indices, channel_indices, y_coords, x_coords]
    
    detections = []
    for x, y, score in zip(x_coords, y_coords, scores):
        detections.append((x.item(), y.item(), score.item()))
    return detections

# ── 4. EVALUATION & VISUALIZATION ─────────────────────────────
def run_evaluation():
    # 1. Parse the results.txt file for valid image names
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: Could not find {RESULTS_FILE}")
        return

    with open(RESULTS_FILE, 'r') as f:
        lines = f.readlines()
    
    # Filter out headers and empty lines, keep only .jpg/.JPG lines
    unseen_images = [line.strip() for line in lines if line.strip().lower().endswith('.jpg')]
    
    if len(unseen_images) < NUM_TEST_IMGS:
        print(f"Warning: Found only {len(unseen_images)} images, testing all of them.")
        test_images = unseen_images
    else:
        test_images = random.sample(unseen_images, NUM_TEST_IMGS)
        print(f"Randomly selected {NUM_TEST_IMGS} images from {len(unseen_images)} unseen candidates.")

    # 2. Load Model
    model = SimpleUNetCenterNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    print(f"Loaded weights from: {MODEL_WEIGHTS}\n")

    # 3. Process Images
    for img_filename in test_images:
        base_name = Path(img_filename).stem
        img_path = os.path.join(IMG_DIR, img_filename)
        lbl_path = os.path.join(LBL_DIR, base_name + ".txt")
        
        # Load Image
        orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if orig_img is None: 
            print(f"Failed to load image: {img_path}")
            continue
            
        img = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE))
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(DEVICE)

        # Load Ground Truth
        gt_centroids = []
        if os.path.isfile(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    if line.startswith("1 "):
                        parts = line.strip().split()
                        cx, cy = float(parts[1]), float(parts[2])
                        gt_centroids.append((cx * IMG_SIZE, cy * IMG_SIZE))

        # Model Prediction
        with torch.no_grad():
            pred_heatmap = model(img_tensor)
        
        # Extract Coordinates
        predicted_peaks = extract_peaks(pred_heatmap, threshold=CONF_THRESH)
        
        # --- Plotting (2 Side-by-Side Panels) ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f"Unseen Image Evaluation: {base_name}", fontsize=16)

        # Panel 1: Ground Truth
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(f"Ground Truth ({len(gt_centroids)} MBPs)", fontsize=14)
        for gx, gy in gt_centroids:
            axes[0].plot(gx, gy, 'go', markersize=8, fillstyle='none', markeredgewidth=2) # Green circles

        # Panel 2: Predicted Centroids
        axes[1].imshow(img, cmap='gray')
        axes[1].set_title(f"Prediction ({len(predicted_peaks)} MBPs, Conf >= {CONF_THRESH})", fontsize=14)
        for px, py, score in predicted_peaks:
            axes[1].plot(px, py, 'rx', markersize=8, markeredgewidth=2) # Red crosses

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        
        # Save directly to the model's directory
        out_filename = os.path.join(MODEL_DIR, f"unseen_test_{base_name}.png")
        plt.savefig(out_filename, dpi=150)
        plt.close()
        print(f"Saved: {out_filename}")

if __name__ == "__main__":
    run_evaluation()