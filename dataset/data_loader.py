import torch
from torch_geometric.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .cifar import Cifar100
from .graph_generator import GraphDataGenerator

import random
import numpy as np


class CifarGraphDataset(Dataset):
    def __init__(self, data_folder="./DATA/CIFARv2Mini",
                       phase="train", 
                       n_segments=50, 
                       compactness=30, 
                       connectivity=2, 
                       n_node_features=97):

        super().__init__()

        cifar100 = Cifar100(data_dir=data_folder, phase=phase)
        images, image_names, image_labels = cifar100.load_images()

        data_generator = GraphDataGenerator(images=images, image_names=image_names, image_labels=image_labels)

        self.data_list = data_generator.generate_graph_dataset(n_segments=n_segments, 
                                                            compactness=compactness, 
                                                            connectivity=connectivity, 
                                                            n_node_features=n_node_features)
        # print(len(self.data_list), self.data_list[0].x.size(), self.data_list[0].edge_index.size(), self.data_list[0].x.dtype)

    def len(self):
        return len(self.data_list)

    def get(self, index):
        return self.data_list[index]
    

class CifarGraphLoader:
    def __init__(self, data_folder="./DATA/CIFARv2Mini",
                       phase="train",
                       batch_size=128,
                       shuffle=True,
                       random_seed=42,
                       test_size=0.2):
        self.data_folder = data_folder
        self.phase = phase
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.test_size = test_size

    def load_data(self):

        train_dataset = CifarGraphDataset(data_folder=self.data_folder, phase="train")
        test_dataset = CifarGraphDataset(data_folder=self.data_folder, phase="test")


        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

        test_loader = DataLoader(
            dataset=test_dataset, batch_size=1, shuffle=False
        )

        return (train_loader, test_loader)
