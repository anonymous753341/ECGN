import os
import sys
sys.path.append("/home/ec2-user/ECGN")
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler
import scipy.stats as stats
from dgl.dataloading import NeighborSampler, MultiLayerFullNeighborSampler
from dgl.dataloading import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, explained_variance_score
import dgl
import random
from sklearn.metrics._regression import r2_score
from main.pytorchtools import EarlyStopping
from dgl.nn import SAGEConv
import SMOTE.utils as utils
import SMOTE.models as models
from train_clusters.baseline_clusters.baseline_cluster_model import BaselineGraphSAGECluster, GTModel, SparseMHA
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
class BaselineTrainClustersSMOTEImbalancedOriginal:

    def __init__(self, g_whole, opt, labels_whole, device):
        super(BaselineTrainClustersSMOTEImbalancedOriginal, self).__init__()
        self.g_whole = g_whole
        self.opt = opt
        self.labels_whole = labels_whole
        self.device = device

    
    def train_cluster_gnn_baselines(self, idx_synthetic_new, g_whole_temp, cluster_nodes, cluster_model, cluster_optimizer, feature_encoder, optimizer_feature_encoder, classifier, optimizer_cls, train_indices, val_indices, test_indices, new_node_to_cluster_map, original_train_nodes):
        
        early_stopping = EarlyStopping(patience=30, verbose=True)
        device = self.device


        # g_whole_temp = g_whole_temp.to(device)
        train_indices = torch.tensor(train_indices)
        cluster_nodes = torch.tensor(cluster_nodes)

        train_mask = torch.isin(train_indices, cluster_nodes)
        cluster_specific_indices_train = train_indices[train_mask]


        original_cluster_nodes = [node for node in cluster_nodes if node in torch.cat([original_train_nodes, val_indices, test_indices])]
        original_cluster_nodes = torch.stack(original_cluster_nodes, dim=0)#.tolist()


        ch_mask = torch.isin(cluster_nodes, original_cluster_nodes)
        cluster_specific_indices = cluster_nodes[ch_mask]

        train_check_mask = torch.isin(cluster_nodes, cluster_specific_indices_train)
        # cluster_specific_indices_train = train_indices[train_mask]

         # Create training subgraph and move to device
        # cluster_nodes = cluster_nodes.to(device)
        cluster_graph = g_whole_temp.subgraph(cluster_nodes).to(device)
        cluster_labels = cluster_graph.ndata['label'].to(device)
        cluster_features = cluster_graph.ndata['feat'].to(device)

        for e in range(self.opt.cluster_epochs):
            cluster_model.train()
            classifier.train()

            # Forward pass on the training indices
            with autocast():  # Mixed precision training
                logits_train = cluster_model(cluster_graph, cluster_features)[train_check_mask]
                logits_train = classifier(logits_train)
                loss_train = F.cross_entropy(logits_train, cluster_labels[train_check_mask])
            
            early_stopping(loss_train, cluster_model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
            cluster_optimizer.zero_grad()
            optimizer_cls.zero_grad()
            scaler.scale(loss_train).backward()#loss_train.backward()
            scaler.step(cluster_optimizer)#cluster_optimizer.step()
            scaler.step(optimizer_cls)#optimizer_cls.step()
            scaler.update()
            
        # original_cluster_nodes = cluster_nodes[torch.tensor([node in torch.cat([original_train_nodes, val_indices, test_indices]) for node in cluster_nodes])]
            
        # Evaluation
        cluster_model.eval()
        # classifier.eval()

        # Generate latent embeddings
        with torch.no_grad(), autocast():
            latent_embeddings_all = cluster_model(cluster_graph, cluster_features)#[ch_mask]

        return cluster_model, latent_embeddings_all, original_cluster_nodes


    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.manual_seed(42)
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, dgl.nn.SAGEConv):
            torch.manual_seed(42)
            if m.fc_neigh is not None:
                torch.nn.init.xavier_uniform_(m.fc_neigh.weight)
            if m.fc_self is not None:
                torch.nn.init.xavier_uniform_(m.fc_self.weight)
                torch.nn.init.zeros_(m.fc_self.bias)
    
    def smooth_embeddings(self, g, embeddings, index_map):
        # Create a copy of the graph to avoid modifying the original graph
        g = g.local_var()

        # Ensure the embeddings tensor is on the same device as the graph
        embeddings = embeddings.to(g.device)
        
        # Assign the new embeddings to the nodes specified in index_map
        g.ndata['feat'][index_map] = embeddings

        # Compute the mean of neighbor embeddings
        g.update_all(
            message_func=dgl.function.copy_u(u='feat', out='m'),
            reduce_func=dgl.function.mean(msg='m', out='feat')
        )

        # Retrieve the smoothed embeddings
        smoothed_embeddings = g.ndata['feat'][index_map]

        return smoothed_embeddings
    # Define the init_weights function
    # def init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.manual_seed(42)
    #         nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
    #     elif isinstance(m, nn.BatchNorm1d):
    #         nn.init.ones_(m.weight)
    #         nn.init.zeros_(m.bias)
    #     elif isinstance(m, SparseMHA):
    #         torch.manual_seed(42)
    #         for param in m.parameters():
    #             if param.dim() > 1:
    #                 nn.init.xavier_uniform_(param)
    #             else:
    #                 nn.init.zeros_(param)

    def initialize_model(self, in_dim):
        model = BaselineGraphSAGECluster(in_dim, self.opt.layerdim, self.opt.aggregator, in_dim, self.opt.layer, self.opt.dropout).to(device=self.device)
        # model = BaselineGraphSAGECluster(in_dim, in_dim, self.opt.aggregator, in_dim, self.opt.layer, self.opt.dropout).to(device=self.device)
        # model =  GTModel(in_dim, self.opt.layerdim, self.opt.layerdim, self.opt.layer, 2)
        model.apply(self.init_weights)
        return model

    
    def process_clusters(self, train_indices, val_indices, test_indices, node_to_cluster_map, train=True):
        H_clusters = []
        index_map_processing = []

        neighborhood_sampling= self.opt.neighborhood_sampling

        cluster_count = 0


        smoteEarlier=False

        if smoteEarlier:
            #Apply SMOTE before training

            # Extract adjacency matrix
            adj = self.g_whole.adj_external()

            # Extract node features
            features = self.g_whole.ndata['feat'].float()

            # Extract node labels
            labels = self.g_whole.ndata['label']

            # labels = torch.where(labels == 0, torch.tensor(1), torch.tensor(0))

            #Updated adjacency matrix extraction

            adj_new, embed, labels_new, idx_train_new, idx_synthetic_new, new_node_to_cluster_map = utils.src_smote_original_distance_cluster(adj, features, labels, train_indices, node_to_cluster_map, portion=self.opt.im_ratio, im_class_num=3)

            #Get new temporal graph object with SMOTE

            adj_graph = adj_new.to_dense().numpy()

            src, dst = np.nonzero(adj_graph)
            g_whole_temp = dgl.graph((src, dst))

            # Add node features and labels
            g_whole_temp.ndata['feat'] = embed
            g_whole_temp.ndata['label'] = labels_new

            idx_train_new = idx_train_new.tolist()
           
        else:

            g_whole_temp = self.g_whole
            idx_train_new = train_indices.tolist()
            idx_synthetic_new = torch.tensor([])
            new_node_to_cluster_map = node_to_cluster_map
            labels_new = self.g_whole.ndata['label']


        for cluster in set(new_node_to_cluster_map.values()):

            cluster_count = cluster_count + 1
            cluster_nodes = [node for node, clust in new_node_to_cluster_map.items() if clust == cluster]
            original_cluster_nodes = [node for node, clust in node_to_cluster_map.items() if clust == cluster]
            in_dim = g_whole_temp.ndata['feat'].shape[-1]

            cluster_model = self.initialize_model(in_dim).to(self.device)
            cluster_optimizer = torch.optim.Adam(cluster_model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weightdecay)


            classifier = models.Classifier(nembed= in_dim, 
                                           nhid= self.opt.layerdim,
                                           nclass=len(labels_new.unique()),
                                           dropout=0.0).to(self.device)

            optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=self.opt.lr, weight_decay=self.opt.weightdecay)
            
            feature_encoder= "EMPTY"
            optimizer_feature_encoder = "EMPTY"



            if train:
                if neighborhood_sampling:
                    cluster_model, latent_embeddings, cluster_specific_indices= self.train_cluster_gnn_baselines(idx_synthetic_new, g_whole_temp, cluster_nodes, cluster_model, cluster_optimizer, feature_encoder, optimizer_feature_encoder, classifier, optimizer_cls, idx_train_new, val_indices, test_indices, new_node_to_cluster_map, train_indices)
                else:
                    # g_whole_temp = g_whole_temp.to(self.device)
                    cluster_model, latent_embeddings, cluster_specific_indices= self.train_cluster_gnn_baselines(idx_synthetic_new, g_whole_temp, cluster_nodes, cluster_model, cluster_optimizer, feature_encoder, optimizer_feature_encoder, classifier, optimizer_cls, idx_train_new, val_indices, test_indices, new_node_to_cluster_map, train_indices)

            # cluster_mask = torch.tensor([node in original_cluster_nodes for node in range(self.g_whole.number_of_nodes())], dtype=torch.bool)
            # latent_space = latent_embeddings[:cluster_mask.shape[0]][cluster_mask] #latent_embeddings[cluster_mask]

            H_clusters.append(latent_embeddings.to('cpu'))

            # index_map_processing.append(original_cluster_nodes)
            index_map_processing.append(original_cluster_nodes)


            print("Done for Cluster:"+ str(cluster_count))
        

        with torch.no_grad():
            self.g_whole.ndata['feat'] = torch.zeros(self.g_whole.ndata['feat'].shape[0], self.opt.layerdim) #in_dim
            # self.g_whole.ndata['feat'][torch.tensor([item for sublist in index_map_processing for item in sublist])] = torch.cat(H_clusters, dim=0).float()

            # #Smoothing
            concatenated_embeddings =  torch.cat(H_clusters, dim=0).float()

            # Assuming index_map is a tensor mapping subcluster nodes back to the original graph
            index_map = torch.tensor([item for sublist in index_map_processing for item in sublist])

            # Smooth the embeddings
            smoothed_embeddings = self.smooth_embeddings(self.g_whole, concatenated_embeddings, index_map)

            # Put back the smoothed embeddings into the original graph
            self.g_whole.ndata['feat'][index_map] = smoothed_embeddings
            
        return self.g_whole
