from skimage.measure import regionprops
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.future import graph

from typing import List

class GraphDataGenerator:
    def __init__(self, images: List, connectivity=2):

        self.images = images

        self.segmentation_slic = []
        self.img_graphs = []
        self.img_props = []


    def image_to_graph(image, n_segments=50, compactness=30, connectivity=2):
        segments = slic(image, n_segments=n_segments, compactness=compactness)
        g = graph.RAG(segments, connectivity=connectivity)
        props = regionprops(segments, image)
        return segments, g, props


    def generate_graph():
        for i in self.images:
            segments, g, props = generate_graph(i, n_segments)
            segmentation_slic.append(segments)
            img_graphs.append(g)
            img_props.append(props)
        

    def encode_edges(g):
        E = [[], []]
        for i, j in g.edges():
            E[0].append(i-1)
            E[0].append(j-1)
            E[1].append(j-1)
            E[1].append(i-1)
        return E


    def create_edges():
        edges = []
        for i in range(len(img_graphs)):
            edge_index = encode_edges(img_graphs[i])
            edges.append(edge_index)
        return edges
len(edges)