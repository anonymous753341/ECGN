import sys
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn import SAGEConv #GraphConv as SAGEConvunder
from torch_geometric.nn import TransformerConv
import torch



class GraphSAGENeighbor(nn.Module):
    def __init__(self, in_feats, h_feats, aggregator, num_classes, num_layers, dropout=0.5, num_active_layers=2):
        super(GraphSAGENeighbor, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        # self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(0.0)
        
        # Input layer
        self.convs.append(SAGEConv(in_feats, h_feats, aggregator_type=aggregator))
        # self.batch_norms.append(nn.BatchNorm1d(h_feats))
        
        # Hidden layers
        for _ in range(1, num_layers - 1):
            self.convs.append(SAGEConv(h_feats, h_feats, aggregator_type=aggregator))
            # self.batch_norms.append(nn.BatchNorm1d(h_feats))
        
        # Output layer
        self.convs.append(SAGEConv(h_feats, num_classes, aggregator_type=aggregator))
        # self.batch_norms.append(nn.BatchNorm1d(num_classes))
    
    def forward(self, mfgs, x):
        h = x
        for i in range(self.num_layers):
            h_dst = h[:mfgs[i].num_dst_nodes()]
            h = self.convs[i](mfgs[i], (h, h_dst))
            # h = self.batch_norms[i](h)  # Apply batch norm
            if i != self.num_layers - 1:  # No activation or dropout on the output layer
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def freeze_layers(self):
        num_layer_to_activate = self.num_active_layers
        if self.num_layers == 1 or self.num_layers == 2:
            for param in self.convs[0].parameters():
                param.requires_grad = False
        elif self.num_layers > 2:
            # Freeze all layers except the last two (or as specified by num_active_layers)
            for i in range(len(self.convs) - num_layer_to_activate):
                for param in self.convs[i].parameters():
                    param.requires_grad = False
            # Ensure the last few layers are trainable
            for i in range(len(self.convs) - num_layer_to_activate, len(self.convs)):
                for param in self.convs[i].parameters():
                    param.requires_grad = True
    
    def get_optimizer(self, base_lr=0.01, multiplier=0.01, weight_decay=1e-3):
        lr = base_lr
        params = []
        num_layers = len(self.convs)

        for i in range(num_layers):
            if i < num_layers - 1:
                params.append({'params': self.convs[i].parameters(), 'lr': lr * multiplier})
            else:
                params.append({'params': self.convs[i].parameters(), 'lr': lr})

        optimizer = torch.optim.Adam(params, weight_decay=weight_decay)
        return optimizer

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, aggregator, num_classes, num_layers, dropout=0.5, num_active_layers=2):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.num_active_layers = num_active_layers

        # Create graph convolutional layers dynamically based on num_layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_feats=in_feats, out_feats=h_feats, aggregator_type=aggregator, activation=None))
        for _ in range(1, num_layers):
            self.convs.append(SAGEConv(in_feats=h_feats, out_feats=h_feats, aggregator_type=aggregator, activation=None))
        
        # Final convolutional layer
        self.final_conv = SAGEConv(in_feats=h_feats, out_feats=num_classes, aggregator_type=aggregator, activation=None)
    
    def forward(self, g, in_feat):
        in_feat = in_feat.float()
        h = in_feat
        for conv in self.convs:
            h = conv(g, h)
            h = F.relu(h)
            h = self.dropout(h)
        h = self.final_conv(g, h)
        return h
    
    def freeze_layers(self):
        num_layer_to_activate = self.num_active_layers
        if self.num_layers == 1 or self.num_layers == 2:
            for param in self.convs[0].parameters():
                param.requires_grad = False
        elif self.num_layers > 2:
            # Freeze all layers except the last two
            for i in range(len(self.convs) - num_layer_to_activate):
                for param in self.convs[i].parameters():
                    param.requires_grad = False
            # Ensure the last two layers are trainable
            for i in range(len(self.convs) - num_layer_to_activate, len(self.convs)):
                for param in self.convs[i].parameters():
                    param.requires_grad = True
            # Ensure the final layer is trainable
            for param in self.final_conv.parameters():
                param.requires_grad = True
    
    # Function to set up layer-wise learning rates with smaller rates for initial layers
    def get_optimizer(self, base_lr=0.01, multiplier=0.01, weight_decay=1e-03):
        lr = base_lr
        params = []
        num_layers = len(self.convs) #+ 1  # Including final_conv

        for i in range(num_layers):
            if i < len(self.convs)-1:
                params.append({'params': self.convs[i].parameters(), 'lr': lr * multiplier})
            else:
                params.append({'params': self.convs[i].parameters(), 'lr': lr})
            # lr *= multiplier  # Increase learning rate for the next layer

        params.append({'params': self.final_conv.parameters(), 'lr': lr})
        optimizer = torch.optim.Adam(params, weight_decay=weight_decay)
        return optimizer



class GraphSAGEClusterBlocks(nn.Module):
    def __init__(self, in_feats, h_feats, aggregator, num_classes, num_layers, dropout=0.5):
        super(GraphSAGEClusterBlocks, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_feats, h_feats, aggregator_type=aggregator))
        self.batch_norms.append(nn.BatchNorm1d(h_feats))

        # Hidden layers
        for _ in range(1, num_layers - 1):
            self.convs.append(SAGEConv(h_feats, h_feats, aggregator_type=aggregator))
            self.batch_norms.append(nn.BatchNorm1d(h_feats))

        # Output layer
        self.convs.append(SAGEConv(h_feats, num_classes, aggregator_type=aggregator))
        self.batch_norms.append(nn.BatchNorm1d(num_classes))

    def forward(self, blocks, x):
        h = x.float()
        for l_id, (layer, block) in enumerate(zip(self.convs, blocks)):
            h = layer(block, h)
            if l_id != len(self.convs) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h