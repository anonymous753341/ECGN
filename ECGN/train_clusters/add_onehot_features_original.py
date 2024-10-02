import os
import sys
sys.path.append("/home/ec2-user/ECGN")
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
# import EarlyStopping
from sklearn.metrics._regression import r2_score
from dgl.nn import SAGEConv #GraphConv as SAGEConvunder
import SMOTE.utils as utils
import SMOTE.models as models
from train_clusters.cluster_model import GraphSAGECluster


def add_onehot_features(clusters, g_whole ):

    # Initialize lists to store small clusters and mappings
    small_clusters = []
    index_map = []
    label_encodings = []
    label_count = 0

    # Dictionary to map nodes to their corresponding cluster index
    node_to_cluster_map = {}

    # Iterate through clusters
    for cluster_idx, cluster in enumerate(clusters):
        # cluster = torch.tensor(list(cluster))
        # cluster = torch.tensor(list(cluster_set))
        if cluster.numel() > 15:
            index_map.append(cluster)
            repeated_value_list = [label_count] * cluster.numel()
            repeated_value_tensor = torch.tensor(repeated_value_list)
            label_encodings.append(repeated_value_tensor)
            label_count += 1
            # Map nodes to cluster index
            for node in cluster:
                node_to_cluster_map[node.item()] = cluster_idx
        else:
            # cluster = torch.tensor(list(cluster))
            small_clusters.append(cluster)

    # Handle small clusters if any
    if small_clusters:
        small_clusters = [t.unsqueeze(0) if t.dim() == 0 else t for t in small_clusters]
        combined_cluster = torch.cat(small_clusters, dim=0)
        index_map.append(combined_cluster)
        label_count_final = len(label_encodings) + 1
        repeated_value_list = [label_count_final] * combined_cluster.shape[0]
        repeated_value_tensor = torch.tensor(repeated_value_list)
        label_encodings.append(repeated_value_tensor)
        # Map nodes to combined small clusters index
        combined_model_index = len(index_map) - 1
        for node in combined_cluster:
            node_to_cluster_map[node.item()] = combined_model_index

    # Convert label_encodings to one-hot encoding
    label_encodings = [t.unsqueeze(0) if t.dim() == 0 else t for t in label_encodings]
    label_encodings = torch.cat(label_encodings, dim=0)
    max_label = int(torch.max(label_encodings).item())
    one_hot_encodings = torch.zeros(label_encodings.size(0), max_label + 1)
    one_hot_encodings.scatter_(1, label_encodings.unsqueeze(1), 1)

    # Ensure index_map is properly formatted
    index_map_final = [t.unsqueeze(0) if t.dim() == 0 else t for t in index_map]
    index_map_final = torch.cat(index_map_final, dim=0)

    # Sort index_map_final and apply sorting to labels
    sorted, sorted_indices = torch.sort(index_map_final)
    all_additional_feature = one_hot_encodings[sorted_indices]

    # Assuming g_whole.ndata['feat'] contains original node features
    original_feature = g_whole.ndata['feat']

    # Concatenate original features with additional features
    g_whole.ndata['feat'] = torch.cat((original_feature, all_additional_feature), dim=1)

    return g_whole, index_map, node_to_cluster_map

def filter_indices(index_map, indices, node_to_cluster_map):
    filtered_index_map = []
    cluster_numbers = []
    for cluster_idx, cluster_tensor in enumerate(index_map):
        filtered_indices = [idx for idx in cluster_tensor if idx in indices]
        if filtered_indices:
            filtered_index_map.append(torch.tensor(filtered_indices))
            specific_cluster_number = node_to_cluster_map[torch.tensor(filtered_indices)[0].item()]
            cluster_numbers.append(specific_cluster_number)
    return filtered_index_map, cluster_numbers