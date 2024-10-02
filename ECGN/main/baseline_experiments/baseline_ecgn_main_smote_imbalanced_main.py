import os
import sys


sys.path.append("/home/ec2-user/ECGN")
from SMOTE import models
from train_clusters.baseline_clusters.baseline_cluster_model import BaselineGraphSAGECluster

from ogb.nodeproppred import DglNodePropPredDataset
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
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
from baseline_global_model import GraphSAGE, GraphSAGEClusterBlocks
from train_clusters.training_clusters import TrainClusters
from train_clusters.add_onehot_features import add_onehot_features, filter_indices
from datasets import data_load, utils
from train_clusters.locality_hashing import locality_hashing_clusters
from train_clusters.metis_partitioning import metis_clusters
from train_clusters.baseline_clusters.training_clusters_smote_imbalanced import BaselineTrainClustersSMOTEImbalanced
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
wandb.init(project='uncategorized', save_code=True, config="/home/ec2-user/ECGN/sweeps/ogbn_replication/baseline_cluster_sweep_sage_imbalanced_ogbn.yaml")
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
    dataset = CoraGraphDataset()
    g_whole = dataset[0]
    labels = g_whole.ndata['label']
    features = g_whole.ndata['feat']


    train_mask = g_whole.ndata['train_mask']
    val_mask = g_whole.ndata['val_mask']
    test_mask = g_whole.ndata['test_mask']

    train_indices, val_indices, test_indices =  utils.split_arti(labels, majority_class_size=200, minority_class_size=20) #utils.split_arti(labels, 0.3)
    adj = g_whole.adj_external().to_dense()

elif dataset_name == 'citeseer':
    dataset = CiteseerGraphDataset()
    g_whole = dataset[0]
    labels = g_whole.ndata['label']
    features = g_whole.ndata['feat']


    train_mask = g_whole.ndata['train_mask']
    val_mask = g_whole.ndata['val_mask']
    test_mask = g_whole.ndata['test_mask']

    train_indices, val_indices, test_indices = utils.split_arti(labels, majority_class_size=200, minority_class_size=20)
    adj = g_whole.adj_external().to_dense()


elif dataset_name == 'Reddit':
    dataset = RedditDataset()
    g_whole = dataset[0]
    labels = g_whole.ndata['label']
    features = g_whole.ndata['feat']

    train_mask = g_whole.ndata['train_mask']
    val_mask = g_whole.ndata['val_mask']
    test_mask = g_whole.ndata['test_mask']

    # train_indices, val_indices, test_indices = utils.split_arti(labels, 0.3)

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
    train_indices, val_indices, test_indices = utils.split_arti(labels, majority_class_size=800, minority_class_size=100)

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


elif dataset_name=="ogbn_arxiv":

    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()

    g_whole, labels = dataset[0]

    features = g_whole.ndata['feat']

    num_nodes = features.shape[0]

    labels = torch.tensor(labels.squeeze())

    # adj = g_whole.adj_external().to_dense()
    train_indices, val_indices, test_indices = utils.split_arti(labels, majority_class_size=800, minority_class_size=40, num_of_imbalanced_classes= 10)

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

    g_whole.ndata['label'] = labels




if opt.cluster_epochs > 0:

    # clusters = locality_hashing_clusters(g = g_whole, feature_matrix = features, train_indices = train_indices, num_clusters=opt.num_clusters)
    clusters = metis_clusters(g = g_whole, feature_matrix = features, train_indices = train_indices, num_clusters=10)


    #############################Adding one hot encoded cluster embedding features ####################

    g_whole, index_map, node_to_cluster_map = add_onehot_features(clusters, g_whole=g_whole)
    original_features_with_clusters = g_whole.ndata['feat'].to(device)

    baselineSmote=False

    if baselineSmote:

        #Smote at beginning
        #Intra Cluster SMOTE: src_smote_original_distance_cluster
        #Oversample: 
        #SMOTE: src_smote
        #Upsample: src_upsample
        adj_new, embed, labels, train_indices = smoteutils.src_upsample(adj, features, labels, train_indices,  im_class_num=3)
        # adj_new, embed, labels, train_indices, idx_synthetic_new, new_node_to_cluster_map = smoteutils.src_smote_original_distance_cluster(adj, features, labels, train_indices, node_to_cluster_map, portion=opt.im_ratio, im_class_num=10)


        adj_graph = adj_new.to_dense().numpy()

        src, dst = np.nonzero(adj_graph)
        g_whole = dgl.graph((src, dst))

        # Add node features and labels
        g_whole.ndata['feat'] = embed
        g_whole.ndata['label'] = labels

    else:

        _trainClusters = BaselineTrainClustersSMOTEImbalanced(g_whole, opt, labels, device)
        # Train clusters for the training set and update embeddings
        # g_whole, train_indices, labels= _trainClusters.process_clusters(train_indices, val_indices, test_indices, node_to_cluster_map, train=True)
        g_whole = _trainClusters.process_clusters(train_indices, val_indices, test_indices, node_to_cluster_map, train=True)



torch.cuda.empty_cache()

input_dim = g_whole.ndata['feat'].shape[-1]
nclass = torch.max(g_whole.ndata['label']).item()+1


opt.dropout=0.0

if opt.cluster_epochs > 0 and opt.layer_wise_lr!=True :
         
    model = GraphSAGE(input_dim, opt.layerdim, opt.aggregator, nclass, opt.layer, opt.dropout, opt.num_active_layers).to(device)
    model.freeze_layers()

elif opt.cluster_epochs > 0 and opt.layer_wise_lr==True :
    model = GraphSAGE(input_dim, opt.layerdim, opt.aggregator, nclass, opt.layer, opt.dropout, opt.num_active_layers).to(device)
        # model.freeze_layers()
else:
    model = GraphSAGE(input_dim, opt.layerdim, opt.aggregator, nclass, opt.layer, opt.dropout, opt.num_active_layers).to(device)
print(model)



if opt_method == 'Adam':
    if opt.cluster_epochs > 0 and opt.layer_wise_lr==True:
        optimizer = model.get_optimizer(base_lr = opt.lr, multiplier = 1, weight_decay=opt.weightdecay)
    elif opt.cluster_epochs > 0 and opt.layer_wise_lr==False:
        optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr , weight_decay=opt.weightdecay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr , weight_decay=opt.weightdecay)
    
elif opt_method == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr * 0.01, momentum = 0.8, weight_decay=opt.weightdecay, nesterov = True)

fullbatch_training = opt.fullbatch

warnings.filterwarnings("ignore", category=UserWarning) 
if fullbatch_training:

    ######### Training together ###########
    # Move the whole graph and features to the device
    g_whole = g_whole.to(device)
    features = g_whole.ndata['feat'].to(device) #original_features_with_clusters 
    labels = labels.to(device)

    # Training loop with validation and testing
    training_losses = []
    validation_losses = []
    testing_losses = []

    for e in range(opt.n_epochs):
        model.train()
        logits_train = model(g_whole, features)[train_indices]
        loss_train = F.cross_entropy(logits_train, labels[train_indices])
        preds_train = logits_train.argmax(dim=1)
        correct_train = (preds_train == labels[train_indices]).sum().item()
        accuracy_train = correct_train / len(labels[train_indices])
        precision_train = precision_score(labels[train_indices].cpu(), preds_train.cpu(), average='macro', zero_division=0)
        recall_train = recall_score(labels[train_indices].cpu(), preds_train.cpu(), average='macro', zero_division=0)
        f1_train = f1_score(labels[train_indices].cpu(), preds_train.cpu(), average='macro', zero_division=0)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # Store training losses
        training_losses.append(loss_train.item())

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            logits_val = model(g_whole, features)[val_indices]
            # logits_val = classifier(logits_val)
            loss_val = F.cross_entropy(logits_val, labels[val_indices])
            preds_val = logits_val.argmax(dim=1)
            correct_val = (preds_val == labels[val_indices]).sum().item()
            accuracy_val = correct_val / len(labels[val_indices])
            precision_val = precision_score(labels[val_indices].cpu(), preds_val.cpu(), average='macro', zero_division=0)
            recall_val = recall_score(labels[val_indices].cpu(), preds_val.cpu(), average='macro', zero_division=0)
            f1_val = f1_score(labels[val_indices].cpu(), preds_val.cpu(), average='macro', zero_division=0)

        # Store validation losses
        validation_losses.append(loss_val.item())

        # Evaluate on test set
        with torch.no_grad():
            logits_test = model(g_whole, features)[test_indices]
            # logits_test = classifier(logits_test)
            loss_test = F.cross_entropy(logits_test, labels[test_indices])
            preds_test = logits_test.argmax(dim=1)
            correct_test = (preds_test == labels[test_indices]).sum().item()
            accuracy_test = correct_test / len(labels[test_indices])
            precision_test = precision_score(labels[test_indices].cpu(), preds_test.cpu(), average='macro', zero_division=0)
            recall_test = recall_score(labels[test_indices].cpu(), preds_test.cpu(), average='macro', zero_division=0)
            f1_test = f1_score(labels[test_indices].cpu(), preds_test.cpu(), average='macro', zero_division=0)

        # Store testing losses
        testing_losses.append(loss_test.item())

        # Print training, validation, and testing losses, accuracies, precision, recall, and F1 scores
        if e % 5 == 0:
            print('Epoch {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Testing Loss: {:.4f}, '
                  'Training Accuracy: {:.4f}, Validation Accuracy: {:.4f}, Testing Accuracy: {:.4f}, '
                  'Training Precision: {:.4f}, Validation Precision: {:.4f}, Testing Precision: {:.4f}, '
                  'Training Recall: {:.4f}, Validation Recall: {:.4f}, Testing Recall: {:.4f}, '
                  'Training F1 Score: {:.4f}, Validation F1 Score: {:.4f}, Testing F1 Score: {:.4f}'.format(
                    e, loss_train.item(), loss_val.item(), loss_test.item(),
                    accuracy_train, accuracy_val, accuracy_test,
                    precision_train, precision_val, precision_test,
                    recall_train, recall_val, recall_test,
                    f1_train, f1_val, f1_test))

            # Log the metrics to wandb
            wandb.log({
                'metis': "True",
                'Epoch': e,
                'Training Loss': loss_train.item(),
                'Validation Loss': loss_val.item(),
                'Testing Loss': loss_test.item(),
                'Training Accuracy': accuracy_train,
                'Validation Accuracy': accuracy_val,
                'Testing Accuracy': accuracy_test,
                'Training Precision': precision_train,
                'Validation Precision': precision_val,
                'Testing Precision': precision_test,
                'Training Recall': recall_train,
                'Validation Recall': recall_val,
                'Testing Recall': recall_test,
                'Training F1 Score': f1_train,
                'Validation F1 Score': f1_val,
                'Testing F1 Score': f1_test
            })

    ########### End training together ##########
else:

    # # Training, validation, and test graphs
    g_train_sub = g_whole.subgraph(train_mask)#.to(device)
    g_val_sub = g_whole.subgraph(val_mask)#.to(device)
    g_test_sub = g_whole.subgraph(test_mask)#.to(device)

    # # Labels for training, validation, and test sets
    labels_train_sub = labels[train_mask].to(device)
    labels_val_sub = labels[val_mask].to(device)
    labels_test_sub = labels[test_mask].to(device)


    print("Training:" + str(torch.bincount(labels_train_sub.long())))
    print("Validation:" + str(torch.bincount(labels_val_sub.long())))
    print("Testing:" + str(torch.bincount(labels_test_sub.long())))

    torch.cuda.empty_cache()


    def train( g, labels, batch_size):
        model.train()
        num_nodes = g.num_nodes()
        total_loss = 0
        total_correct = 0
        torch.cuda.empty_cache()

        all_preds = []
        all_labels = []



        # # Shuffle nodes
        shuffled_nodes = torch.randperm(num_nodes)

        for batch_start in range(0, num_nodes, batch_size):
            
            # Get a batch of node indices
            batch_nodes = shuffled_nodes[batch_start:batch_start+batch_size]

            # Create a subgraph containing only the batch nodes
            batched_graph = dgl.node_subgraph(g, batch_nodes).to(device)

            # Load features and labels for the batch nodes
            batch_features = g.ndata['feat'][batch_nodes].to(device)
            batch_labels = labels[batch_nodes]

            # Forward pass
            logits_train = model(batched_graph, batch_features)

            # logp_train = F.log_softmax(logits_train, 1)
            # loss = F.nll_loss(logp_train, batch_labels)
            loss = F.cross_entropy(logits_train, batch_labels)

            total_loss += loss.item()
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update total correct predictions
            preds_train = logits_train.argmax(dim=1)
            correct_train = (preds_train == batch_labels).sum().item()
            total_correct += correct_train
            # Store predictions and labels for precision, recall, and F1 score calculation
            all_preds.extend(preds_train.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

        accuracy = total_correct / num_nodes
        avg_loss = total_loss / num_nodes

        # Calculate precision, recall, and F1 score
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return avg_loss, accuracy, precision, recall, f1

    def evaluate_model( g, labels, batch_size):
        model.eval()
        num_nodes = g.num_nodes()
        total_loss = 0
        total_correct = 0

        all_preds = []
        all_labels = []


        # Shuffle nodes
        shuffled_nodes = torch.randperm(num_nodes)
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch_start in range(0, num_nodes, batch_size):
                # Get a batch of node indices
                batch_nodes = shuffled_nodes[batch_start:batch_start+batch_size]
                # Create a subgraph containing only the batch nodes
                batched_graph = dgl.node_subgraph(g, batch_nodes).to(device)

                # Load features and labels for the batch nodes
                batch_features = g.ndata['feat'][batch_nodes].to(device)
                batch_labels = labels[batch_nodes]


                # Forward pass
                logits_test = model(batched_graph, batch_features)
                # logp_test = F.log_softmax(logits_test, 1)
                # loss = F.nll_loss(logp_test, batch_labels)
                loss = F.cross_entropy(logits_test, batch_labels)
                total_loss += loss.item()


                # Update total correct predictions
                preds_test = logits_test.argmax(dim=1)
                correct_test = (preds_test == batch_labels).sum().item()
                total_correct += correct_test

            # Store predictions and labels for precision, recall, and F1 score calculation
            all_preds.extend(preds_test.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

        accuracy = total_correct / num_nodes
        avg_loss = total_loss / num_nodes

        # Calculate precision, recall, and F1 score
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")

        return avg_loss, accuracy, precision, recall, f1


    early_stopping = EarlyStopping(patience=20, verbose=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    for epoch in range(0, num_epoch):
        since  = time.time()

    
        # Train and evaluate the model
        loss_train, acc_train, prec_train, rec_train, f1_train = train(g_train_sub, labels_train_sub, batch_size=opt.batchSize)
        loss_val, acc_val, prec_val, rec_val, f1_val = evaluate_model(g_val_sub, labels_val_sub, batch_size=opt.batchSize )
        loss_test, acc_test, prec_test, rec_test, f1_test = evaluate_model(g_test_sub, labels_test_sub, batch_size=opt.batchSize)
        
        early_stopping(loss_val, model)
            
        if early_stopping.early_stop:
            print("Early stopping")
            break

        time_elapsed = time.time() - since
        print('*====**')

        if loss_val < best_loss and epoch > 5:
            print("saving best model")
            best_loss = loss_val
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_model:
                torch.save(best_model_wts, os.path.join(opt.save_path, str(fold) + 'working_model_subsample' + '.pth'))
        print("Epoch:", epoch,
            "Training Loss:", loss_train,
            "Training Accuracy:", acc_train,
            "Training Precision:", prec_train,
            "Training Recall:", rec_train,
            "Training F1-Score:", f1_train,
            "Validation Loss:", loss_val,
            "Validation Accuracy:", acc_val,
            "Validation Precision:", prec_val,
            "Validation Recall:", rec_val,
            "Validation F1-Score:", f1_val,
            "Testing Loss:", loss_test,
            "Testing Accuracy:", acc_test,
            "Testing Precision:", prec_test,
            "Testing Recall:", rec_test,
            "Testing F1-Score:", f1_test,
            "Time Elapsed:", time_elapsed)

