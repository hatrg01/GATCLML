import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from .cifar import Cifar100
from .graph_generator import GraphDataGenerator

import os
import random
import numpy as np
import pickle

from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import matplotlib.pyplot as plt


class CifarGraphDataset(Dataset):
    def __init__(self, data_folder="./DATA/CIFARv2Mini",
                       phase="train", 
                       n_segments=50, 
                       compactness=30, 
                       connectivity=2, 
                       n_node_features=97,
                       pre_data_list_dir='./GRAPHDATA/graph_data_list'):

        super().__init__()

        self.data_list = []
        self.pre_data_list_dir = f"{pre_data_list_dir}_{phase}.pkl"

        if os.path.exists(self.pre_data_list_dir):
            print(f"LOADING GRAPH DATA FROM DIR : {self.pre_data_list_dir} !")
            self.data_list = pickle.load(open(self.pre_data_list_dir, 'rb'))
        else:
            cifar100 = Cifar100(data_dir=data_folder, phase=phase)
            images, image_names, image_labels = cifar100.load_images()

            data_generator = GraphDataGenerator(images=images, image_names=image_names, image_labels=image_labels)

            self.data_list = data_generator.generate_graph_dataset(n_segments=n_segments, 
                                                                compactness=compactness, 
                                                                connectivity=connectivity, 
                                                                n_node_features=n_node_features)
            pickle.dump(self.data_list, open(f"/content/drive/MyDrive/RESEARCH/DATA/graph_data_list_{phase}.pkl", 'wb'))
            print("SAVING GRAPH DATA !")


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


# if __name__ == '__main__':
#     # loader = CifarGraphLoader()
#     # train_loader, test_loader = loader.load_data()

#     # print("Number of batch in train loader : ", len(train_loader))
#     # print("Number of batch in test loader : ", len(test_loader))

#     # for step, data in enumerate(train_loader):
#     #     print(f'Step {step + 1}:')
#     #     print('=======')
#     #     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     #     print(data)
#     #     print()

#     data_list = pickle.load(open("../GRAPHDATA/graph_data_list_train.pkl", 'rb'))

#     D = data_list[1000]

#     print(D)

#     # print(D.edge_index)

#     G = to_networkx(D, to_undirected=True)
#     # nx.draw(G)


#     # G = nx.Graph()
#     # G.add_node(1)
#     # G.add_nodes_from([2, 3])
#     # G.add_nodes_from([
#     #     (4, {"color": "red"}),
#     #     (5, {"color": "green"}),
#     # ])
#     # G.add_edge(1, 2)
#     # e = (2, 3)
#     # G.add_edge(*e)

#     nx.draw(G, with_labels=True, font_weight='bold')
#     # plt.show(block=False)
#     plt.savefig("Graph.png", format="PNG")


