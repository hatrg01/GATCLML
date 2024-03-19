import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class GATCLML(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_classes, num_heads, concat=True, dropout=0.2):
        super(GATCLML, self).__init__()

        self.gat = nn.Sequential(
        GATConv(in_features, hidden_dim, heads=num_heads, concat=True, dropout=dropout),
        GATConv(num_heads * hidden_dim, hidden_dim, heads=num_heads, concat=True, dropout=dropout),
        GATConv(num_heads * hidden_dim, hidden_dim, heads=1, concat=False)
        )

        self.metric_learning = nn.Sequential(
        nn.Linear(hidden_dim, 512),
        nn.ELU(),
        nn.Dropout(dropout),
        # nn.Linear(1024, 256),
        # nn.ELU(),
        # nn.Dropout(dropout),
        nn.Linear(512, out_features),
        nn.ELU()
        )

        self.classifier = nn.Linear(out_features, num_classes)


    def forward(self, x, edge_index, batch):
        for layer in self.gat:
            x = layer(x, edge_index)
            x = F.elu(x)
        h = global_max_pool(x, batch=batch)
        embedding = self.metric_learning(h.view(h.size(0), -1))
        logits = self.classifier(embedding)
        return embedding, logits

