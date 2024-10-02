import os
import sys
sys.path.append("/home/ec2-user/ECGN")
from SMOTE import models
from train_clusters.baseline_clusters.baseline_cluster_model import BaselineGraphSAGECluster
from train_clusters.baseline_clusters.training_clusters_smote_imbalanced_nofinetune import BaselineTrainClustersSMOTEImbalancedNoFineTune
import numpy as np
import argparse
import time
import copy
import torch
import torch.nn.functional as F
from pytorchtools  import EarlyStopping
import random
from sklearn.metrics._regression import r2_score
import wandb
import dgl
from baseline_global_model import GraphSAGE, GraphSAGEClusterBlocks
from train_clusters.training_clusters import TrainClusters
from train_clusters.add_onehot_features import add_onehot_features, filter_indices
from datasets import data_load, utils
from train_clusters.locality_hashing import locality_hashing_clusters
from train_clusters.metis_partitioning import metis_clusters
from train_clusters.baseline_clusters.training_clusters_smote_imbalanced import BaselineTrainClustersSMOTEImbalanced
from train_clusters.baseline_clusters.training_clusters_smote_imbalanced_original import BaselineTrainClustersSMOTEImbalancedOriginal
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from dgl.data import CiteseerGraphDataset, AmazonCoBuyComputerDataset, RedditDataset, CoraGraphDataset, CoraFullDataset
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
import SMOTE.utils as smoteutils

# from dgl.dataloading import ClusterGCNClusterData

torch.manual_seed(123)
EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['WANDB_DISABLE_CODE'] = 'false'
wandb.init(project='uncategorized', save_code=True, config="/home/ec2-user/ECGN/sweeps/baseline_cluster_sweep_sage_imbalanced.yaml")
#wandb.init(project='abcd_fluid_prediction_publish', save_code=True)
config = wandb.config
torch.manual_seed(1)
np.random.seed(1111)
random.seed(1111)
torch.cuda.manual_seed_all(1111)

torch.manual_seed(2)
parser = argparse.ArgumentParser()
parser.add_argument('--stepsize', type=int, default=30, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.1, help='scheduler shrinking rate')
parser.add_argument('--indim', type=int, default=1188, help='feature dim')
parser.add_argument('--nclass', type=int, default=7, help='num of classes')
parser.add_argument('--fold', type=int, default=0, help='training which fold')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')

#Arguments from WANDB Sweeps

parser.add_argument('--lr', type = float, default=config.lr, help='learning rate')
parser.add_argument('--epoch', type=int, default=config.epoch, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=config.n_epochs, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=config.batchSize, help='size of the batches')
parser.add_argument('--weightdecay', type=float, default=config.weightdecay, help='regularization')
parser.add_argument('--layer', type=int, default=config.layer, help='number of GNN layers')
parser.add_argument('--optim', type=str, default=config.optim, help='optimization method: SGD, Adam')
parser.add_argument('--early_stop_steps', type=int, default=config.early_stop_steps, help='Early Stopping Steps')
parser.add_argument('--layerdim', type=int, default=config.layerdim, help='dimension of hidden layers')
parser.add_argument('--aggregator', type=str, default=config.aggregator, help='aggregation strategy for graphsage')
parser.add_argument('--cluster_epochs', type=int, default=config.cluster_epochs, help='cluster training number of epochs')

parser.add_argument('--num_clusters', type=int, default=config.num_clusters, help='Number of clusters using LSH')
parser.add_argument('--num_active_layers', type=int, default=config.num_active_layers, help='Number of active layers')

parser.add_argument('--neighborhood_sampling', type=bool, default=config.neighborhood_sampling, help='neighborhood or subsample training')
parser.add_argument('--im_ratio', type=float, default=config.im_ratio, help='imbalanced upscale ratio')

parser.add_argument('--dropout', type=float, default=config.dropout, help='dropout')

parser.add_argument('--layer_wise_lr', type=bool, default=config.layer_wise_lr, help='enable layer wise lr')
parser.add_argument('--datasetname', type=str, default=config.datasetname, help='dataset name')
parser.add_argument('--fullbatch', type=bool, default=config.fullbatch, help='fullbatch')
opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

#################### Parameter Initialization #######################
path = "."
dataset_name = opt.datasetname
save_model = opt.save_model
load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
fold = opt.fold



#####Loading the datasets ######
# Load data
if dataset_name == 'cora':
    adj, features, labels = data_load.load_data()
    
    adj_graph = adj.to_dense().numpy()

    src, dst = np.nonzero(adj_graph)
    g_whole = dgl.graph((src, dst))

    # Add node features and labels
    g_whole.ndata['feat'] = features
    g_whole.ndata['label'] = labels
    train_indices, val_indices, test_indices = utils.split_arti(labels, majority_class_size=200, minority_class_size=20)


elif dataset_name == 'citeseer':
    dataset = CiteseerGraphDataset()
    g_whole = dataset[0]
    labels = g_whole.ndata['label']
    features = g_whole.ndata['feat']


    train_mask = g_whole.ndata['train_mask']
    val_mask = g_whole.ndata['val_mask']
    test_mask = g_whole.ndata['test_mask']

    train_indices, val_indices, test_indices = utils.split_arti(labels, majority_class_size=200, minority_class_size=20)


elif dataset_name == 'Reddit':
    dataset = RedditDataset()
    g_whole = dataset[0]
    labels = g_whole.ndata['label']
    features = g_whole.ndata['feat']

    train_mask = g_whole.ndata['train_mask']
    val_mask = g_whole.ndata['val_mask']
    test_mask = g_whole.ndata['test_mask']

    train_indices = train_mask.nonzero(as_tuple=False).squeeze()
    val_indices = val_mask.nonzero(as_tuple=False).squeeze()
    test_indices = test_mask.nonzero(as_tuple=False).squeeze()

elif dataset_name == 'AmazonComputer':
    dataset = AmazonCoBuyComputerDataset()
    g_whole = dataset[0]
    labels = g_whole.ndata['label']
    features = g_whole.ndata['feat']

    num_nodes = features.shape[0]

    adj = g_whole.adj_external().to_dense()
    train_indices, val_indices, test_indices = utils.split_arti(labels, majority_class_size=200, minority_class_size=20)

    # Initialize masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Set the corresponding indices to True
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Optionally, assign the masks to the graph's node data
    g_whole.ndata['train_mask'] = train_mask
    g_whole.ndata['val_mask'] = val_mask
    g_whole.ndata['test_mask'] = test_mask


    train_indices, val_indices, test_indices, class_num_mat = utils.split_genuine(labels)



if opt.cluster_epochs > 0:

    clusters = metis_clusters(g = g_whole, feature_matrix = features, train_indices = train_indices, num_clusters=3)


    #############################Adding one hot encoded cluster embedding features ####################

    g_whole, index_map, node_to_cluster_map = add_onehot_features(clusters, g_whole=g_whole)
    original_features_with_clusters = g_whole.ndata['feat'].to(device)

    baselineSmote=False

    if baselineSmote:

        #Smote at beginning

        adj_new, embed, labels, train_indices, idx_synthetic_new, new_node_to_cluster_map = smoteutils.src_smote_original(adj, features, labels, train_indices, node_to_cluster_map, portion=opt.im_ratio, im_class_num=3)

        # train_indices = idx_train_new.copy()
        #Get new temporal graph object with SMOTE

        adj_graph = adj_new.to_dense().numpy()

        src, dst = np.nonzero(adj_graph)
        g_whole = dgl.graph((src, dst))

        # Add node features and labels
        g_whole.ndata['feat'] = embed
        g_whole.ndata['label'] = labels

    else:

        _trainClusters = BaselineTrainClustersSMOTEImbalancedNoFineTune(g_whole, opt, labels, device)
        # Train clusters for the training set and update embeddings
        metrics= _trainClusters.process_clusters(train_indices, val_indices, test_indices, node_to_cluster_map, train=True)
        print(metrics)


