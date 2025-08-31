import os
import random
import pickle
import argparse
from os import listdir
from os.path import isfile, isdir, join
from collections import defaultdict as ddict
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split




class SkinConDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

        self.data['concept_values'] = self.data['concept_values'].apply(eval)  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.root_dir, row['ImageID'])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        concept_values = torch.tensor(row['concept_values'], dtype=torch.float32)
        label = torch.tensor(row['three_partition_label'], dtype=torch.long)
        
        return image, label, concept_values