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
from train_clusters.baseline_clusters.baseline_cluster_model import BaselineGraphSAGECluster, LargeBaselineGraphSAGECluster, GTModel, SparseMHA
import copy

class ProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.proj = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.proj(x)
    
class BaselineTrainClustersSMOTEImbalancedLargeGraphs:

    def __init__(self, g_whole, opt, labels_whole, device):
        super(BaselineTrainClustersSMOTEImbalancedLargeGraphs, self).__init__()
        self.g_whole = g_whole
        self.opt = opt
        self.labels_whole = labels_whole
        self.device = device

    def train_cluster_gnn_baselines_largegraphs(self, idx_synthetic_new, g_whole_temp, cluster_nodes, cluster_model, cluster_optimizer, feature_encoder, optimizer_feature_encoder, classifier, optimizer_cls, train_indices, val_indices, test_indices, new_node_to_cluster_map, original_train_nodes):

        device = self.device

        # Combine training and validation indices to use all data for training
        # all_train_indices = torch.tensor(train_indices + val_indices.tolist())#.to(self.device)
        all_train_indices = torch.tensor(train_indices)#.to(self.device)
        cluster_nodes = torch.tensor(cluster_nodes)#.to(device)

        train_mask = torch.isin(all_train_indices, cluster_nodes)
        cluster_specific_indices_train = all_train_indices[train_mask]

        original_cluster_nodes = [node for node in cluster_nodes if node in torch.cat([original_train_nodes, val_indices, test_indices])]
        original_cluster_nodes = torch.stack(original_cluster_nodes, dim=0)#.to(device)

        ch_mask = torch.isin(cluster_nodes, original_cluster_nodes)
        cluster_specific_indices = cluster_nodes[ch_mask]

        train_check_mask = torch.isin(cluster_nodes, cluster_specific_indices_train)

        sampler_layers = []
        opt_layer= [4,4,4,4]

        # Construct the sampler based on opt.layer
        for indices in range(self.opt.layer):
            sampler_layers.append(opt_layer[indices])
        
        sampler = dgl.dataloading.NeighborSampler(sampler_layers)
        # sampler = MultiLayerFullNeighborSampler(1)

        # Create training subgraph and move to device
        cluster_graph = g_whole_temp.subgraph(cluster_nodes)#.to(device)
        cluster_labels = cluster_graph.ndata['label'].to(device)

        # all_labels = g_whole_temp.ndata['label'].to(device)

        train_dataloader = DataLoader(cluster_graph, torch.arange(0, len(cluster_nodes))[train_check_mask], sampler,  batch_size=self.opt.batchSize, shuffle=True, num_workers=0, device=self.device, use_uva=True)
        # train_dataloader = DataLoader(g_whole_temp, torch.tensor(cluster_nodes)[train_check_mask], sampler,  batch_size=self.opt.batchSize, shuffle=True, num_workers=0, device=self.device, use_uva=True)



        best_model_wts = copy.deepcopy(cluster_model.state_dict())
        best_loss = 1e10

        early_stopping = EarlyStopping(patience=30, verbose=False)

        for e in range(self.opt.cluster_epochs):
            # Training phase
            cluster_model.train()
            classifier.train()
            
            for input_nodes, output_nodes, blocks in train_dataloader:
                cluster_optimizer.zero_grad()
                optimizer_cls.zero_grad()
                
                logits_train = cluster_model(blocks, blocks[0].srcdata['feat'])
                logits_train = classifier(logits_train)
                labels_new = cluster_labels[output_nodes].long()
                # labels_new = all_labels[output_nodes].long()
                loss_train = F.cross_entropy(logits_train, labels_new)

                # Perform the training step
                cluster_optimizer.zero_grad()
                optimizer_cls.zero_grad()
                loss_train.backward()
                cluster_optimizer.step()
                optimizer_cls.step()

                early_stopping(loss_train, cluster_model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                # Track the best model based on training loss
                if loss_train < best_loss and e > 5:
                    print("saving best model")
                    best_loss = loss_train
                    best_model_wts = copy.deepcopy(cluster_model.state_dict())

        # Load the best model weights
        cluster_model.load_state_dict(best_model_wts)

        # Evaluation on the entire subgraph
        cluster_model.eval()

        all_dataloader = DataLoader(cluster_graph, torch.arange(0, len(cluster_nodes)), sampler,  batch_size=self.opt.batchSize, shuffle=True, num_workers=0, device=self.device, use_uva=True)
        # all_dataloader = DataLoader(g_whole_temp, torch.tensor(cluster_nodes), sampler,  batch_size=self.opt.batchSize, shuffle=True, num_workers=0, device=self.device, use_uva=True)
        latent_embeddings_list = []
        latent_embeddings_list = []
        output_nodes_list = []
        
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in all_dataloader:
                try:
                    # Ensure blocks and features are on the correct device
                    blocks = [block.to(self.device) for block in blocks]
                    input_features = blocks[0].srcdata['feat'].to(self.device)
                    embed = cluster_model(blocks, input_features)
                    latent_embeddings_list.append(embed)
                    output_nodes_list.append(output_nodes)
                    
                except Exception as e:
                    print(f'Error in evaluation loop: {e}')
                    raise
            
        # Concatenate lists to tensors
        latent_embeddings = torch.cat(latent_embeddings_list, dim=0)
        output_nodes = torch.cat(output_nodes_list, dim=0)


        return cluster_model, latent_embeddings, output_nodes
    
    def train_cluster_gnn_baselines(self, idx_synthetic_new, g_whole_temp, cluster_nodes, cluster_model, cluster_optimizer, feature_encoder, optimizer_feature_encoder, classifier, optimizer_cls, train_indices, val_indices, test_indices, new_node_to_cluster_map, original_train_nodes):

        device = self.device

        # Combine training and validation indices to use all data for training
        # all_train_indices = torch.tensor(train_indices + val_indices.tolist())#.to(self.device)
        all_train_indices = torch.tensor(train_indices)#.to(self.device)
        cluster_nodes = torch.tensor(cluster_nodes)#.to(device)

        train_mask = torch.isin(all_train_indices, cluster_nodes)
        cluster_specific_indices_train = all_train_indices[train_mask]

        original_cluster_nodes = [node for node in cluster_nodes if node in torch.cat([original_train_nodes, val_indices, test_indices])]
        original_cluster_nodes = torch.stack(original_cluster_nodes, dim=0)#.to(device)

        ch_mask = torch.isin(cluster_nodes, original_cluster_nodes)
        cluster_specific_indices = cluster_nodes[ch_mask]

        train_check_mask = torch.isin(cluster_nodes, cluster_specific_indices_train)

        # Create training subgraph and move to device
        cluster_graph = g_whole_temp.subgraph(cluster_nodes).to(device)
        cluster_labels = cluster_graph.ndata['label'].to(device)
        cluster_features = cluster_graph.ndata['feat'].to(device)

        best_model_wts = copy.deepcopy(cluster_model.state_dict())
        best_loss = 1e10

        early_stopping = EarlyStopping(patience=30, verbose=False)
        for e in range(self.opt.cluster_epochs):
            # Training phase
            cluster_model.train()
            classifier.train()

            # Forward pass on the full dataset
            logits_train = cluster_model(cluster_graph, cluster_features)[train_check_mask]
            logits_train = classifier(logits_train)
            loss_train = F.cross_entropy(logits_train, cluster_labels[train_check_mask])

            # Perform the training step
            cluster_optimizer.zero_grad()
            optimizer_cls.zero_grad()
            loss_train.backward()
            cluster_optimizer.step()
            optimizer_cls.step()

            early_stopping(loss_train, cluster_model)

            if early_stopping.early_stop:
                # print("Early stopping")
                break

            # Track the best model based on training loss
            if loss_train < best_loss and e > 5:
                # print("saving best model")
                best_loss = loss_train
                best_model_wts = copy.deepcopy(cluster_model.state_dict())

        # Load the best model weights
        cluster_model.load_state_dict(best_model_wts)

        # Evaluation on the entire subgraph
        cluster_model.eval()
        with torch.no_grad():
            latent_embeddings_all = cluster_model(cluster_graph, cluster_features)

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


    def initialize_model(self, in_dim):
        # model = BaselineGraphSAGECluster(in_dim, self.opt.layerdim , self.opt.aggregator, in_dim, 1, self.opt.dropout).to(device=self.device)
        model = LargeBaselineGraphSAGECluster(in_dim, self.opt.layerdim , self.opt.aggregator, in_dim, self.opt.layer, self.opt.dropout).to(device=self.device)
        # model = BaselineGraphSAGECluster(in_dim, in_dim, self.opt.aggregator, in_dim, self.opt.layer, self.opt.dropout).to(device=self.device)
        # model =  GTModel(in_dim, self.opt.layerdim, self.opt.layerdim, self.opt.layer, 2)
        model.apply(self.init_weights)
        return model

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


    
    def process_clusters(self, train_indices, val_indices, test_indices, node_to_cluster_map, train=True):
        H_clusters = []
        index_map_processing = []
        cluster_weights_list = []  # List to store weights for each cluster

        neighborhood_sampling= self.opt.neighborhood_sampling

        cluster_count = 0


        smote=False

        if smote:
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

            # feature_encoder = models.Sage_En(nfeat=self.g_whole.ndata['feat'].shape[-1],
            #                                   nhid=self.opt.layerdim,
            #                                   nembed=self.opt.layerdim,
            #                                   dropout=0.2, num_layers=self.opt.layer).to(self.device)
            # optimizer_feature_encoder = torch.optim.Adam(feature_encoder.parameters(), lr=self.opt.lr, weight_decay=self.opt.weightdecay)
            cluster_count = cluster_count + 1
            cluster_nodes = [node for node, clust in new_node_to_cluster_map.items() if clust == cluster]
            original_cluster_nodes = [node for node, clust in node_to_cluster_map.items() if clust == cluster]
            in_dim = g_whole_temp.ndata['feat'].shape[-1]

            cluster_model = self.initialize_model(in_dim).to(self.device)
            


            classifier = models.Classifier(nembed= in_dim, 
                                           nhid= self.opt.layerdim,  #self.opt.layerdim
                                           nclass=len(labels_new.unique()),
                                           dropout=0.0).to(self.device)

            optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=self.opt.lr, weight_decay=self.opt.weightdecay)

            cluster_optimizer = torch.optim.Adam(cluster_model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weightdecay)

            # cluster_optimizer = torch.optim.Adam(list(cluster_model.parameters()) + list(classifier.parameters()), lr=self.opt.lr, weight_decay=self.opt.weightdecay)
            
            feature_encoder= "EMPTY"
            optimizer_feature_encoder = "EMPTY"



            if train:
                if neighborhood_sampling:
                    cluster_model, latent_embeddings, cluster_specific_indices= self.train_cluster_gnn_baselines_largegraphs(idx_synthetic_new, g_whole_temp, cluster_nodes, cluster_model, cluster_optimizer, feature_encoder, optimizer_feature_encoder, classifier, optimizer_cls, idx_train_new, val_indices, test_indices, new_node_to_cluster_map, train_indices)
                else:
                    # g_whole_temp = g_whole_temp.to(self.device)
                    cluster_model, latent_embeddings, cluster_specific_indices= self.train_cluster_gnn_baselines_largegraphs(idx_synthetic_new, g_whole_temp, cluster_nodes, cluster_model, cluster_optimizer, feature_encoder, optimizer_feature_encoder, classifier, optimizer_cls, idx_train_new, val_indices, test_indices, new_node_to_cluster_map, train_indices)

            # cluster_mask = torch.tensor([node in original_cluster_nodes for node in range(self.g_whole.number_of_nodes())], dtype=torch.bool)
            # latent_space = latent_embeddings[:cluster_mask.shape[0]][cluster_mask] #latent_embeddings[cluster_mask]

            # Store the weights in memory
            cluster_weights_list.append(cluster_model.state_dict())

            H_clusters.append(latent_embeddings)

            # index_map_processing.append(original_cluster_nodes)
            index_map_processing.append(original_cluster_nodes)


            # print("Done for Cluster:"+ str(cluster_count))
        

        with torch.no_grad():
            new_features = torch.zeros(self.g_whole.ndata['feat'].shape[0], self.opt.layerdim) #self.opt.layerdim

            # Replace the existing features with the new zero features
            self.g_whole.ndata['feat'] = new_features
            # self.g_whole.ndata['feat'][torch.tensor([item for sublist in index_map_processing for item in sublist])] = torch.cat(H_clusters, dim=0).float()
            
            # #Smoothing
            concatenated_embeddings =  torch.cat(H_clusters, dim=0).float().to('cpu')

            # Assuming index_map is a tensor mapping subcluster nodes back to the original graph
            index_map = torch.tensor([item for sublist in index_map_processing for item in sublist])

            # Smooth the embeddings
            smoothed_embeddings = self.smooth_embeddings(self.g_whole, concatenated_embeddings, index_map)

            # Put back the smoothed embeddings into the original graph
            self.g_whole.ndata['feat'][index_map] = smoothed_embeddings

            # #Average the weights

            # averaged_weights = cluster_weights_list[0]
            # for key in averaged_weights.keys():
            #     for i in range(1, len(cluster_weights_list)):
            #         averaged_weights[key] += cluster_weights_list[i][key]
            #     averaged_weights[key] /= len(cluster_weights_list)
           
            # global_model = self.initialize_model(in_dim).to(self.device)
            # global_model.load_state_dict(averaged_weights)

            #Test smote here

            # adj = self.g_whole.adj_external()
            # features = self.g_whole.ndata['feat'].float()
            # labels = self.g_whole.ndata['label']
            # adj_new, embed, labels_new, idx_train_new, idx_synthetic_new, new_node_to_cluster_map = utils.src_smote_original_distance_cluster(adj, features, labels, train_indices, node_to_cluster_map, portion=self.opt.im_ratio, im_class_num=5)
            # adj_graph = adj_new.to_dense().numpy()

            # src, dst = np.nonzero(adj_graph)
            # g_whole_temp = dgl.graph((src, dst))

            # # Add node features and labels
            # g_whole_temp.ndata['feat'] = embed
            # g_whole_temp.ndata['label'] = labels_new

            # idx_train_new = idx_train_new.tolist()
        return  self.g_whole#, idx_train_new, labels_new #self.g_whole 
