import pandas as pd
import torch
from torch.utils.data import Dataset
#import numpy as np
#import torch.nn as nn

class AD_Dataset(Dataset):
    def __init__(self, data_frame, label_mapping, transform=None):
        self.transform = transform
        self.df = data_frame
         # Check unique values in the 'label' column before mapping
        #print(f"Unique labels before mapping:{ self.df['label'].unique()}")

        # Apply binary label mapping to the 'label' column
        self.df['label'] = self.df['label'].map(label_mapping)

        # Verify label mapping (should  contain: 0, 1, 2)
        unique_labels = self.df['label'].unique()
        if not all(label in [0, 1, 2] for label in unique_labels):
            raise ValueError(f'Label inconsistency after mapping: {unique_labels}')
        #print(f"Unique labels after mapping:{ self.df['label'].unique()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load the image array and binary label from the DataFrame
        image = self.df.iloc[idx]['image'].transpose()
        label = self.df.iloc[idx]['label']

        #print(f'image shape {image.shape}')

        # Convert image to tensor, add channel dimension, and apply transformations if any
        image = torch.tensor(image, dtype=torch.float32)  # Shape (1, height, width)
        label = torch.tensor(label, dtype=torch.long)  # Ensure label is long type for classification

        #print(f'image tensor size: {image.size()}')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label