import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch.nn as nn

class HeartDataset(Dataset):
    def __init__(self, filepaths_pattern, target_shape=(128, 128, 128)):
        self.filepaths = glob.glob(filepaths_pattern)
        self.target_shape = target_shape
        # print(f"Found {len(self.filepaths)} files.") 
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        # print(f"Processing file: {filepath}")  
        try:
            mri_data = self.load_mri(filepath)
            preprocessed_data = self.preprocess_brain_data(mri_data, self.target_shape)
            coords, colors = self.generate_training_data(preprocessed_data)
            return coords, colors
        except Exception as e:
            # print(f"Error processing file {filepath}: {e}")
            return torch.tensor([]), torch.tensor([])
    
    def load_mri(self, filepath):
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            return data
        except Exception as e:
            # print(f"Failed to load MRI data from {filepath}: {e}")
            raise
    
    def preprocess_brain_data(self, data, target_shape):
        try:
            resized_data = resize(data, target_shape, anti_aliasing=True)
            normalized_data = (resized_data - np.min(resized_data)) / (np.max(resized_data) - np.min(resized_data))
            return normalized_data
        except Exception as e:
            # print(f"Failed to preprocess MRI data: {e}")
            raise

    def generate_training_data(self, volume):
        try:
            coords = []
            colors = []
            for x in range(volume.shape[0]):
                for y in range(volume.shape[1]):
                    for z in range(volume.shape[2]):
                        coords.append([x / volume.shape[0], y / volume.shape[1], z / volume.shape[2]])
                        colors.append(volume[x, y, z])
            return torch.tensor(coords, dtype=torch.float32), torch.tensor(colors, dtype=torch.float32)
        except Exception as e:
            # print(f"Failed to generate training data: {e}")
            raise

filepaths_pattern = r'NeRF\NeRF\Task02_Heart\imagesTr\*.nii.gz'
dataset = HeartDataset(filepaths_pattern)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


try:
    for i, (coords, colors) in enumerate(dataloader):
        if coords.nelement() == 0 or colors.nelement() == 0:
            print(f"Skipping file at index {i} due to previous error.")
            continue
        print(f"Visualizing data from file index {i}.")
        plt.imshow(colors[0].view(128, 128, 128)[:, :, 64].numpy(), cmap='gray')
        plt.show()
except Exception as e:
    print(f"An error occurred during processing: {e}")
