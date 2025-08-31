import os
import sys
import copy
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image


N_CONCEPTS = 11

SELECTED_CONCEPTS = [
    "Ring",
    "Line",
    "Arc",
    "Corner",
    "Top-Curve",
    "Semicircles",
    "Triangle",
    "Bottom-Curve",
    "Top-Line",
    "Wedge",
    "Bottom-Line"
]

concept_dict = {
    0: ["Ring"],                           
    1: ["Line"],                             
    2: ["Arc", "Line", "Corner", "Top-Curve"],      
    3: ["Semicircles", "Corner"],            
    4: ["Triangle", "Line", "Corner"],      
    5: ["Arc", "Line", "Corner", "Bottom-Curve"],    
    6: ["Ring", "Top-Line"],                
    7: ["Line", "Corner"],                   
    8: ["Ring", "Wedge"],                    
    9: ["Ring", "Bottom-Line"]          
}

label2concept = {
    0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    2: [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    3: [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    4: [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    5: [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    6: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    7: [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    8: [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    9: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

class MNISTWithConcepts(torch.utils.data.Dataset):
    def __init__(self, dataset, concept_dict):
        self.dataset = dataset
        self.concept_dict = concept_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        concepts = torch.tensor(self.concept_dict[label], dtype=torch.float32)
        return image, label, concepts
    
    
class MNIST_M(torch.utils.data.Dataset):
    def __init__(self, root, train, transform=None):
        self.train = train
        self.transform = transform
        if train:
            self.image_dir = os.path.join(root, 'mnist_m_train')
            labels_file = os.path.join(root, "mnist_m_train_labels.txt")
        else:
            self.image_dir = os.path.join(root, 'mnist_m_test')
            labels_file = os.path.join(root, "mnist_m_test_labels.txt")

        with open(labels_file, "r") as fp:
            content = fp.readlines()
        self.mapping = list(map(lambda x: (x[0], int(x[1])), [c.strip().split() for c in content]))

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        image, label = self.mapping[idx]
        image = os.path.join(self.image_dir, image)
        image = self.transform(Image.open(image).convert('RGB'))
        return image, label


class MNISTDataset:
	"""
	MNIST Dataset class
	"""

	def __init__(self, name, img_dir, LDS_type, is_target):
		self.name = name
		self.img_dir = img_dir
		self.LDS_type = LDS_type
		self.is_target = is_target

	def get_data(self):
		mean, std = 0.5, 0.5
		normalize_transform = transforms.Normalize((mean,), (std,))
		self.train_transforms = transforms.Compose([
								   transforms.ToTensor(),
								   normalize_transform
							   ])
		self.test_transforms = transforms.Compose([
								   transforms.ToTensor(),
								   normalize_transform
								])

		self.train_dataset = datasets.MNIST(self.img_dir, train=True, download=True)
		self.val_dataset = datasets.MNIST(self.img_dir, train=True, download=True)
		self.test_dataset = datasets.MNIST(self.img_dir, train=False, download=True)
		self.train_dataset.name, self.val_dataset.name, self.test_dataset.name = 'DIGITS','DIGITS', 'DIGITS'
		self.num_classes = 10
		return self.num_classes, self.train_dataset, self.val_dataset, self.test_dataset, self.train_transforms, self.test_transforms


class SVHNDataset:
	"""
	SVHN Dataset class

	"""
	def __init__(self, name, img_dir, LDS_type, is_target):
		self.name = name
		self.img_dir = img_dir
		self.LDS_type = LDS_type
		self.is_target = is_target

	def get_data(self):
		mean, std = 0.5, 0.5
		normalize_transform = transforms.Normalize((mean,), (std,))
		RGB2Gray = transforms.Lambda(lambda x: x.convert('L'))
		self.train_transforms = transforms.Compose([
							   RGB2Gray,
							   transforms.Resize((28, 28)),
							   transforms.ToTensor(),
							   normalize_transform
						   ])
		self.test_transforms = transforms.Compose([
							   RGB2Gray,
							   transforms.Resize((28, 28)),
							   transforms.ToTensor(),
							   normalize_transform
						   ])

		self.train_dataset = datasets.SVHN(self.img_dir, split='train', download=True)
		self.val_dataset = datasets.SVHN(self.img_dir, split='train', download=True)
		self.test_dataset = datasets.SVHN(self.img_dir, split='test', download=True)
		self.train_dataset.targets, self.val_dataset.targets, self.test_dataset.targets = \
										self.train_dataset.labels, self.val_dataset.labels, self.test_dataset.labels
		self.num_classes = 10
		return self.num_classes, self.train_dataset, self.val_dataset, self.test_dataset, self.train_transforms, self.test_transforms


class USPSDataset:
    """
    USPS Dataset class
    """

    def __init__(self, name, img_dir, LDS_type, is_target):
        self.name = name
        self.img_dir = img_dir
        self.LDS_type = LDS_type
        self.is_target = is_target

    def get_data(self):
        mean, std = 0.5, 0.5
        normalize_transform = transforms.Normalize((mean,), (std,))
        self.train_transforms = transforms.Compose([
                                   transforms.Resize((28, 28)),
                                   transforms.ToTensor(),
                                   normalize_transform
                               ])
        self.test_transforms = transforms.Compose([
                                   transforms.Resize((28, 28)),
                                   transforms.ToTensor(),
                                   normalize_transform
                               ])

        self.train_dataset = datasets.USPS(self.img_dir, train=True, download=True)
        self.val_dataset = datasets.USPS(self.img_dir, train=True, download=True)
        self.test_dataset = datasets.USPS(self.img_dir, train=False, download=True)
        self.train_dataset.name, self.val_dataset.name, self.test_dataset.name = 'DIGITS','DIGITS', 'DIGITS'
        self.num_classes = 10
        return self.num_classes, self.train_dataset, self.val_dataset, self.test_dataset, self.train_transforms, self.test_transforms
