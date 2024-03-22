import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels):
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = labels == label
            class_embeddings = x[mask]
            center = torch.mean(class_embeddings, dim=0)
            new_centers = self.centers.clone()
            new_centers[label] = center
            self.centers = nn.Parameter(new_centers)

            centers_batch = torch.index_select(self.centers, dim=0, index=labels)
            center_loss = torch.mean(torch.norm(x - centers_batch, p=2, dim=1))
        return self.alpha * center_loss