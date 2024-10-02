import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics._regression import r2_score
from dgl.nn import SAGEConv #GraphConv as SAGEConvunder
import torch
import dgl.sparse as dglsp

class BaselineGraphSAGECluster(nn.Module):
    def __init__(self, in_feats, h_feats, aggregator, num_classes, num_layers, dropout = 0.5):
        super(BaselineGraphSAGECluster, self).__init__()
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        
        # Create graph convolutional layers dynamically based on num_layers
        self.convs = nn.ModuleList()
        # self.convs.append(SAGEConv(in_feats, h_feats, 'none'))
        self.convs.append(SAGEConv(in_feats = in_feats, out_feats = h_feats, aggregator_type= aggregator, activation= None))
        for _ in range(1, num_layers):
            # self.convs.append(SAGEConv(h_feats, h_feats, 'none'))
            self.convs.append(SAGEConv (in_feats = h_feats, out_feats = h_feats, aggregator_type= aggregator, activation= None))
        
    def forward(self, g, in_feat):
        in_feat = in_feat.float()
        h = in_feat
        for conv in self.convs:
            h = conv(g, h)
            h = F.relu(h)
            h = self.dropout(h)
        return h


class LargeBaselineGraphSAGECluster(nn.Module):
    def __init__(self, in_feats, h_feats, aggregator, num_classes, num_layers, dropout=0.5):
        super(LargeBaselineGraphSAGECluster, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_feats, h_feats, aggregator_type=aggregator))
        # self.batch_norms.append(nn.BatchNorm1d(h_feats))

        # Hidden layers
        for _ in range(0, num_layers - 1):
            self.convs.append(SAGEConv(h_feats, h_feats, aggregator_type=aggregator))
            # self.batch_norms.append(nn.BatchNorm1d(h_feats))

        # # Output layer
        # self.convs.append(SAGEConv(h_feats, num_classes, aggregator_type=aggregator))
        # self.batch_norms.append(nn.BatchNorm1d(num_classes))

    def forward(self, blocks, x):
        h = x.float()
        for l_id, (layer, block) in enumerate(zip(self.convs, blocks)):
            h = layer(block, h)
            if l_id != len(self.convs) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h):
        N = len(h)
        # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        # [N, dh, nh]
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)

        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # Sparse softmax by default applies on the last sparse dimension.
        attn = attn.softmax()  # (sparse) [N, N, nh]
        out = dglsp.bspmm(attn, v)  # [N, dh, nh]

        return self.out_proj(out.reshape(N, -1))
     
class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, A, h):
        h1 = h
        h = self.MHA(A, h)
        h = self.batchnorm1(h + h1)

        h2 = h
        h = self.FFN2(F.relu(self.FFN1(h)))
        h = h2 + h

        return self.batchnorm2(h)
    
class GTModel(nn.Module):
    def __init__(
        self,
        in_dim,
        out_size,
        hidden_size=80,
        num_layers=8,
        num_heads=8,
    ):
        super().__init__()
        self.initial_linear = nn.Linear(in_dim, hidden_size)
        # self.pos_linear = nn.Linear(pos_enc_size, hidden_size)
        self.layers = nn.ModuleList(
            [GTLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )
        # self.pooler = dglnn.SumPooling()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, out_size),
        )

    def forward(self, g, X):
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        A = dglsp.spmatrix(indices, shape=(N, N))
        h = self.initial_linear(X)
        # h = X[:,:1438]
        for layer in self.layers:
            h = layer(A, h)
        # h = self.pooler(g, h)

        return self.predictor(h)


