import os
import numpy as np
import pandas as pd
import cv2

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms

# import local modules
from AD_dataset import AD_Dataset
from model import ClassifNetXAI
from gradCAM_func import gradCAM, gradCAMS_saver
from train_func import train_model



DATA_DIR = '/Users/jansta/learn/mri-AD/Data/'

# Load the parquet files
train_data = pd.read_parquet(os.path.join(DATA_DIR, 'train-00000-of-00001-c08a401c53fe5312.parquet'))
test_data = pd.read_parquet(os.path.join(DATA_DIR, 'test-00000-of-00001-44110b9df98c5585.parquet'))

# Convert the 'image' column from bytes to numpy arrays
def convert_to_format(bytes_df):
    img_arr=np.frombuffer(bytes_df,dtype=np.uint8) # frombuffer is used to convert the bytes to np.array
    img=cv2.imdecode(img_arr,cv2.IMREAD_COLOR)    # here cv2 imdecode is used for readblity formate and IMREAD_COLOR is used for rgb color format image 
    return img

def get_img(df):
    images=[]
    for i in range(len(df)):
        inp=df.iloc[i]['image']['bytes']
        cv_data=convert_to_format(inp)
        images.append(cv_data)
    return images

train_data['image'] = get_img(train_data)
test_data['image'] = get_img(test_data) 

label_mapping = {
    0: 0,  # Mild_Demented -> Demented
    1: 0,  # Moderate_Demented -> Demented
    2: 2,  # Non_Demented -> Healthy
    3: 1   # Very_Mild_Demented -> Mild Demented
}

encoded_labels = {
    'Demented': 0,
    'Mild Demented':1,
    'Healthy':2,
}

transform = transforms.Compose([
    #transforms.Resize((64,64)),
    transforms.Grayscale(num_output_channels=1),
    #transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
    ])

train_dataset = AD_Dataset(train_data, label_mapping, transform=transform)
test_dataset = AD_Dataset(test_data, label_mapping, transform=transform)

# Create dataloaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ClassifNetXAI()

NUM_EPOCHS = 300
for rl in [0.00075, 0.0005, 0.00025, 0.00005]:
    out = train_model(model, train_loader, test_loader, encoded_labels, rate_l=rl, NUM_EPOCHS=NUM_EPOCHS,  save=True, thresh=0.5)

