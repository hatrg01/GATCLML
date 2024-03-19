from skimage.measure import regionprops
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.future import graph

from typing import List

import numpy as np

import torch
from torch_geometric.data import Data

from tqdm.auto import tqdm

# np.seterr(divide='ignore', invalid='ignore')


class GraphDataGenerator:
    def __init__(self, images: List, image_names: List, image_labels: List):

        self.images = images
        self.image_names = image_names
        self.image_labels = image_labels


    def image_to_graph(self, image, n_segments=50, compactness=30, connectivity=2):
        segments = slic(image, n_segments=n_segments, compactness=compactness)
        g = graph.RAG(segments, connectivity=connectivity)
        props = regionprops(segments, image)
        return segments, g, props


    def generate_graph(self, n_segments=50, compactness=30, connectivity=2):
        print("Generating RAG!")
        segmentation_slic = []
        img_graphs = []
        img_props = []
        for i in tqdm(self.images):
            segments, g, props = self.image_to_graph(i, n_segments, compactness, connectivity)
            segmentation_slic.append(segments)
            img_graphs.append(g)
            img_props.append(props)
        return segmentation_slic, img_graphs, img_props
        

    def encode_edges(self, g):
        E = [[], []]
        for i, j in g.edges():
            E[0].append(i-1)
            E[0].append(j-1)
            E[1].append(j-1)
            E[1].append(i-1)
        return E


    def generate_edges(self, img_graphs):
        print("Generating edges!")
        edges = []
        for i in tqdm(range(len(img_graphs))):
            edge_index = self.encode_edges(img_graphs[i])
            edges.append(edge_index)
        return edges


    def rag_feature_extraction(self, segmentation, image, n_node_features=97):
        props = regionprops(segmentation, image)
        features = np.zeros((max(np.unique(segmentation)), n_node_features), dtype=np.float32)

        for i, prop in enumerate(props):
            features[i][0:16] = prop['moments'].flatten()

            bbox = prop['bbox']
            features[i][16] = bbox[2] - bbox[0]
            features[i][17] = bbox[3] - bbox[1]

            features[i][18] = prop['area_convex']
            features[i][19] = prop['perimeter']

            features[i][20:23] = prop['intensity_mean']
            features[i][23:26] = prop['intensity_max']
            features[i][26:29] = prop['intensity_min']

            features[i][29:31] = prop['centroid_local']
            features[i][31:37] = prop['centroid_weighted_local'].flatten()

            features[i][37] = prop['orientation']
            features[i][38] = prop['feret_diameter_max']
            features[i][39] = prop['extent']
            features[i][40] = prop['solidity']

            features[i][41:89] = prop['moments_weighted_central'].flatten()
            features[i][89:96] = prop['moments_hu'].flatten()
            features[i][96] = prop['perimeter_crofton']

        return features


    def generate_node_features(self, segmentation_slic, images, n_node_features=97):
        print("Generating node features!")
        node_features = []
        for i in tqdm(range(len(images))):
            feature = self.rag_feature_extraction(segmentation_slic[i], images[i], n_node_features=n_node_features)
            node_features.append(feature)
        return node_features 


    def prepare_data(self, node_features, edges, label):
        x = torch.tensor(node_features)
        edge_index = torch.tensor(edges, dtype=torch.long)
        d = Data(x=x, edge_index=edge_index, y=label)
        return d


    def generate_graph_dataset(self, n_segments=50, compactness=30, connectivity=2, n_node_features=97):

        segmentation_slic, img_graphs, img_props = self.generate_graph(n_segments=n_segments, compactness=compactness, connectivity=connectivity)
        edges = self.generate_edges(img_graphs=img_graphs)
        node_features = self.generate_node_features(segmentation_slic=segmentation_slic, images=self.images, n_node_features=n_node_features)
        
        D = []
        for i in range(len(node_features)):
            d = self.prepare_data(node_features[i], edges[i], self.image_labels[i])
            D.append(d)
        return D