# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

class ClassifNetXAI(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_maps = None
        
        # First Convolutional Block
        self.conv_block1 = nn.Sequential(
            # First convolution: increase number of channels to 16
            nn.Conv2d(1, 16, kernel_size=3, padding=1), #input: (B, 1, H, W) output: (B, 16, H, W)
            nn.BatchNorm2d(16), # BatchNorm to stabilize learning
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # reduces height and width by 2
            
            # Second convolution: further increase channels to 32, also add BatchNorm
            nn.Conv2d(16, 32, kernel_size=3, padding=1), #output: (B, 32, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # reduces height and width by 2

            # Third convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces height and width by 2 -> (B, 64, 8, 8)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (B, 64, 1, 1)

        # Fully connected block
        self.fc_block = nn.Sequential(
            nn.Flatten(),              # Flattens (B, 64, 1, 1) into (B, 64)
            nn.Linear(64, 64),         # Adjusted input features to match the flattened size
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 3)           # 3 == n_classes
        )
    
        
    def forward(self, x: torch.Tensor, store_feature_maps: bool = False) -> torch.Tensor:
        x = self.conv_block1(x)
        #x = self.conv_block2(x)
        
        if store_feature_maps:
            # Detaching feature maps for visualization (e.g., Grad-CAM)
            self.feature_maps = x.detach()
        # Global average pooling: converts (B, 128, H, W) to (B, 128, 1, 1)
        x = self.global_pool(x)
        x = self.fc_block(x)
        # Note: Do not apply an activation like softmax here if you're using CrossEntropyLoss
        return x
