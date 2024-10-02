import argparse
import scipy.sparse as sp
import numpy as np
import torch
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial
import random
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nhid', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='asins')
    parser.add_argument('--size', type=int, default=100)

    parser.add_argument('--epochs', type=int, default=350,
                help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--batch_nums', type=int, default=6000, help='number of batches per epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='number of batches per epoch')


    parser.add_argument('--imbalance', action='store_true', default=True)
    parser.add_argument('--setting', type=str, default='recon', 
        choices=['no','upsampling', 'smote','reweight','embed_up', 'recon','newG_cls','recon_newG'])
    #upsampling: oversample in the raw input; smote: ; reweight: reweight minority classes; 
    # embed_up: 
    # recon: pretrain; newG_cls: pretrained decoder; recon_newG: also finetune the decoder

    parser.add_argument('--opt_new_G', action='store_true', default=False) # whether optimize the decoded graph based on classification result.
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--up_scale', type=float, default=1)
    parser.add_argument('--im_ratio', type=float, default=0.5)
    parser.add_argument('--rec_weight', type=float, default=0.000001)
    parser.add_argument('--model', type=str, default='sage', 
        choices=['sage','gcn','GAT'])



    return parser

def split_arti(labels, c_train_num):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = 25
    c_num_mat[:,2] = 55

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        train_idx = train_idx + c_idx[:c_train_num[i]]
        c_num_mat[i,0] = c_train_num[i]

        val_idx = val_idx + c_idx[c_train_num[i]:c_train_num[i]+25]
        test_idx = test_idx + c_idx[c_train_num[i]+25:c_train_num[i]+80]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat

def split_genuine(labels):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
                ipdb.set_trace()
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/4)
            c_num_mat[i,1] = int(c_num/4)
            c_num_mat[i,2] = int(c_num/2)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat


def print_edges_num(dense_adj, labels):
    c_num = labels.max().item()+1
    dense_adj = np.array(dense_adj)
    labels = np.array(labels)

    for i in range(c_num):
        for j in range(c_num):
            #ipdb.set_trace()
            row_ind = labels == i
            col_ind = labels == j

            edge_num = dense_adj[row_ind].transpose()[col_ind].sum()
            print("edges between class {:d} and class {:d}: {:f}".format(i,j,edge_num))


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def print_class_acc(output, labels, class_num_list, pre='valid'):
    pre_num = 0
    #print class-wise performance
    '''
    for i in range(labels.max()+1):
        
        cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_list[i]
    '''

    #ipdb.set_trace()
    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1).detach(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1)[:,1].detach(), average='macro')

    macro_F = f1_score(labels.detach(), torch.argmax(output, dim=-1).detach(), average='macro')
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    return

# def src_upsample(adj,features,labels,idx_train, portion=1.0, im_class_num=3):
#     c_largest = labels.max().item()
#     # c_largest = 0
#     adj_back = adj.to_dense()
#     chosen = None

#     #ipdb.set_trace()
#     avg_number = int(idx_train.shape[0]/(c_largest+1))

#     for i in range(im_class_num):
#         new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
#         if portion == 0:#refers to even distribution
#             c_portion = int(avg_number/new_chosen.shape[0])

#             for j in range(c_portion):
#                 if chosen is None:
#                     chosen = new_chosen
#                 else:
#                     chosen = torch.cat((chosen, new_chosen), 0)

#         else:
#             c_portion = int(portion)
#             portion_rest = portion-c_portion
#             for j in range(c_portion):
#                 num = int(new_chosen.shape[0])
#                 new_chosen = new_chosen[:num]

#                 if chosen is None:
#                     chosen = new_chosen
#                 else:
#                     chosen = torch.cat((chosen, new_chosen), 0)
            
#             num = int(new_chosen.shape[0]*portion_rest)
#             new_chosen = new_chosen[:num]

#             if chosen is None:
#                 chosen = new_chosen
#             else:
#                 chosen = torch.cat((chosen, new_chosen), 0)
            

#     add_num = chosen.shape[0]
#     new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
#     new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
#     new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
#     new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
#     new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

#     #ipdb.set_trace()
#     features_append = deepcopy(features[chosen,:])
#     labels_append = deepcopy(labels[chosen])
#     idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
#     idx_train_append = idx_train.new(idx_new)

#     features = torch.cat((features,features_append), 0)
#     labels = torch.cat((labels,labels_append), 0)
#     idx_train = torch.cat((idx_train,idx_train_append), 0)
#     adj = new_adj.to_sparse()

#     return adj, features, labels, idx_train

def src_upsample(adj, features, labels, idx_train, portion=1.0,im_class_num=1):
    target_label = 0  # Label to be upsampled
    adj_back = adj.to_dense()
    chosen = None

    new_chosen = idx_train[(labels == target_label)[idx_train]]
    
    if portion == 0:  # Even distribution
        avg_number = int(idx_train.shape[0] / len(torch.unique(labels)))
        c_portion = int(avg_number / new_chosen.shape[0])

        for j in range(c_portion):
            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)

    else:
        c_portion = int(portion)
        portion_rest = portion - c_portion
        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)

        num = int(new_chosen.shape[0] * portion_rest)
        new_chosen = new_chosen[:num]

        if chosen is None:
            chosen = new_chosen
        else:
            chosen = torch.cat((chosen, new_chosen), 0)

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0] + add_num, adj_back.shape[0] + add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:, :]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen, :]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:, chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen, :][:, chosen]

    features_append = deepcopy(features[chosen, :])
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0] + add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train


def src_smote(adj, features, labels, idx_train, node_to_cluster_map, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None
    new_features = None

    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    new_node_to_cluster_map = node_to_cluster_map.copy()

    # # Check if there are enough classes to perform SMOTE
    # if im_class_num > c_largest + 1:
    #     im_class_num = c_largest + 1

    # # If there is no minority class or only one class, return the original data
    # if im_class_num <= 1:
    #     return adj, features, labels, idx_train

    for i in range(im_class_num):

        current_class_label = c_largest - i
        valid_labels = labels[idx_train]
        new_chosen = idx_train[valid_labels == current_class_label]

        # new_chosen = idx_train[(labels == (c_largest - i)).nonzero(as_tuple=True)[0]]
        if new_chosen.shape[0] == 0:
            continue  # Skip if no samples in this class

        if portion == 0:  # refers to even distribution
            c_portion = int(avg_number / new_chosen.shape[0])
            portion_rest = (avg_number / new_chosen.shape[0]) - c_portion
        else:
            c_portion = int(portion)
            portion_rest = portion - c_portion

        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            chosen_embed = features[new_chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)

        num = int(new_chosen.shape[0] * portion_rest)
        new_chosen = new_chosen[:num]

        if num > 0:
            chosen_embed = features[new_chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)

    # If no samples were chosen, return the original data
    if chosen is None:
        return adj, features, labels, idx_train

    add_num = chosen.shape[0]
    new_adj = torch.zeros((adj_back.shape[0] + add_num, adj_back.shape[0] + add_num), device=adj.device)
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen, :]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:, chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen, :][:, chosen]

    features_append = new_features
    labels_append = labels[chosen]
    idx_new = torch.arange(adj_back.shape[0], adj_back.shape[0] + add_num, device=idx_train.device)
    idx_train_append = idx_new

    for original_node, new_node in zip(chosen, idx_new):
        new_node_to_cluster_map[new_node.item()] = node_to_cluster_map[original_node.item()]


    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train, idx_new, new_node_to_cluster_map


# def src_smote_original(adj, features, labels, idx_train, node_to_cluster_map, portion=1.0, im_class_num=3):
#     c_largest = labels.max().item()
#     adj_back = adj.to_dense()
#     chosen = None
#     new_features = None

#     avg_number = int(idx_train.shape[0] / (c_largest + 1))

#     new_node_to_cluster_map = node_to_cluster_map.copy()


#     for i in range(im_class_num):

#         current_class_label = c_largest - i
#         valid_labels = labels[idx_train]
#         new_chosen = idx_train[valid_labels == current_class_label]

#         # new_chosen = idx_train[(labels == (c_largest - i)).nonzero(as_tuple=True)[0]]
#         if new_chosen.shape[0] == 0:
#             continue  # Skip if no samples in this class

#         if portion == 0:  # refers to even distribution
#             c_portion = int(avg_number / new_chosen.shape[0])
#             portion_rest = (avg_number / new_chosen.shape[0]) - c_portion
#         else:
#             c_portion = int(portion)
#             portion_rest = portion - c_portion

#         for j in range(c_portion):
#             num = int(new_chosen.shape[0])
#             chosen_embed = features[new_chosen, :]
#             distance = squareform(pdist(chosen_embed.cpu().detach().numpy()))
#             np.fill_diagonal(distance, distance.max() + 100)

#             idx_neighbor = distance.argmin(axis=-1)
#             interp_place = random.random()
#             embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

#             if chosen is None:
#                 chosen = new_chosen
#                 new_features = embed
#             else:
#                 chosen = torch.cat((chosen, new_chosen), 0)
#                 new_features = torch.cat((new_features, embed), 0)

#         num = int(new_chosen.shape[0] * portion_rest)
#         new_chosen = new_chosen[:num]

#         if num > 0:
#             chosen_embed = features[new_chosen, :]
#             distance = squareform(pdist(chosen_embed.cpu().detach().numpy()))
#             np.fill_diagonal(distance, distance.max() + 100)

#             idx_neighbor = distance.argmin(axis=-1)
#             interp_place = random.random()
#             embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

#             if chosen is None:
#                 chosen = new_chosen
#                 new_features = embed
#             else:
#                 chosen = torch.cat((chosen, new_chosen), 0)
#                 new_features = torch.cat((new_features, embed), 0)

#     # If no samples were chosen, return the original data
#     if chosen is None:
#         return adj, features, labels, idx_train

#     add_num = chosen.shape[0]
#     new_adj = torch.zeros((adj_back.shape[0] + add_num, adj_back.shape[0] + add_num), device=adj.device)
#     new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back
#     new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen, :]
#     new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:, chosen]
#     new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen, :][:, chosen]

#     features_append = new_features
#     labels_append = labels[chosen]
#     idx_new = torch.arange(adj_back.shape[0], adj_back.shape[0] + add_num, device=idx_train.device)
#     idx_train_append = idx_new

#     for original_node, new_node in zip(chosen, idx_new):
#         new_node_to_cluster_map[new_node.item()] = node_to_cluster_map[original_node.item()]


#     features = torch.cat((features, features_append), 0)
#     labels = torch.cat((labels, labels_append), 0)
#     idx_train = torch.cat((idx_train, idx_train_append), 0)
#     adj = new_adj.to_sparse()

#     return adj, features, labels, idx_train, idx_new, new_node_to_cluster_map


def src_smote_original(adj, features, labels, idx_train, node_to_cluster_map, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    chosen = None
    new_features = None

    avg_number = int(idx_train.shape[0] / (c_largest + 1))
    new_node_to_cluster_map = node_to_cluster_map.copy()

    for i in range(im_class_num):
        current_class_label = c_largest - i
        valid_labels = labels[idx_train]
        new_chosen = idx_train[valid_labels == current_class_label]

        if new_chosen.shape[0] == 0:
            continue  # Skip if no samples in this class

        if portion == 0:  # refers to even distribution
            c_portion = int(avg_number / new_chosen.shape[0])
            portion_rest = (avg_number / new_chosen.shape[0]) - c_portion
        else:
            c_portion = int(portion)
            portion_rest = portion - c_portion

        for j in range(c_portion):
            chosen_embed = features[new_chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)

        num = int(new_chosen.shape[0] * portion_rest)
        new_chosen = new_chosen[:num]

        if num > 0:
            chosen_embed = features[new_chosen, :]
            distance = squareform(pdist(chosen_embed.cpu().detach().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed), 0)

    if chosen is None:
        return adj, features, labels, idx_train

    add_num = chosen.shape[0]

    # Update sparse adjacency matrix
    indices = adj._indices()
    values = adj._values()

    new_indices_1 = torch.cat([indices, 
                               torch.cat([chosen.unsqueeze(0).repeat(1, adj.shape[0]), 
                                          torch.arange(adj.shape[0]).unsqueeze(0).repeat(chosen.shape[0], 1).t()], 0)], 1)
    
    new_indices_2 = torch.cat([indices, 
                               torch.cat([torch.arange(adj.shape[0]).unsqueeze(0).repeat(chosen.shape[0], 1).t(), 
                                          chosen.unsqueeze(0).repeat(1, adj.shape[0])], 0)], 1)
    
    new_values = torch.cat([values, adj[chosen, :].to_dense().flatten(), adj[:, chosen].to_dense().flatten()], 0)

    new_adj = torch.sparse.FloatTensor(new_indices_1, new_values, torch.Size([adj.shape[0] + add_num, adj.shape[0] + add_num]))

    features_append = new_features
    labels_append = labels[chosen]
    idx_new = torch.arange(adj.shape[0], adj.shape[0] + add_num, device=idx_train.device)
    idx_train_append = idx_new

    for original_node, new_node in zip(chosen, idx_new):
        new_node_to_cluster_map[new_node.item()] = node_to_cluster_map[original_node.item()]

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)

    return new_adj, features, labels, idx_train, idx_new, new_node_to_cluster_map




def src_smote_original_distance_cluster(adj, features, labels, idx_train, node_to_cluster_map, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None
    new_features = None

    avg_number = int(idx_train.shape[0] / (c_largest + 1))

    new_node_to_cluster_map = node_to_cluster_map.copy()

    for i in range(im_class_num):
        current_class_label = c_largest - i
        valid_labels = labels[idx_train]
        nodes_in_class = idx_train[valid_labels == current_class_label]

        if nodes_in_class.shape[0] == 0:
            continue  # Skip if no samples in this class

        # Compute connectivity with nodes from other clusters
        inter_cluster_connectivity = []
        for node in nodes_in_class:
            cluster_id = node_to_cluster_map[node.item()]
            neighbors = adj_back[node].nonzero(as_tuple=True)[0]
            inter_cluster_edges = sum(node_to_cluster_map[neighbor.item()] != cluster_id for neighbor in neighbors)
            inter_cluster_connectivity.append((node, inter_cluster_edges))
        
        # Sort nodes based on their connectivity to other clusters
        inter_cluster_connectivity.sort(key=lambda x: x[1], reverse=True)
        nodes_with_high_connectivity = torch.tensor([item[0] for item in inter_cluster_connectivity], device=idx_train.device)

        # Choose the top nodes based on the desired portion
        if portion == 0:  # refers to even distribution
            c_portion = int(avg_number / nodes_with_high_connectivity.shape[0])
            portion_rest = (avg_number / nodes_with_high_connectivity.shape[0]) - c_portion
        else:
            c_portion = int(portion)
            portion_rest = portion - c_portion

        for j in range(c_portion):
            num = int(nodes_with_high_connectivity.shape[0])
            chosen_embed = features[nodes_with_high_connectivity, :]
            distance = squareform(pdist(chosen_embed.cpu().detach().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

            if chosen is None:
                chosen = nodes_with_high_connectivity
                new_features = embed
            else:
                chosen = torch.cat((chosen, nodes_with_high_connectivity), 0)
                new_features = torch.cat((new_features, embed), 0)

        num = int(nodes_with_high_connectivity.shape[0] * portion_rest)
        nodes_with_high_connectivity = nodes_with_high_connectivity[:num]

        if num > 0:
            chosen_embed = features[nodes_with_high_connectivity, :]
            distance = squareform(pdist(chosen_embed.cpu().detach().numpy()))
            np.fill_diagonal(distance, distance.max() + 100)

            idx_neighbor = distance.argmin(axis=-1)
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

            if chosen is None:
                chosen = nodes_with_high_connectivity
                new_features = embed
            else:
                chosen = torch.cat((chosen, nodes_with_high_connectivity), 0)
                new_features = torch.cat((new_features, embed), 0)

    # If no samples were chosen, return the original data
    if chosen is None:
        return adj, features, labels, idx_train, None, node_to_cluster_map

    add_num = chosen.shape[0]
    new_adj = torch.zeros((adj_back.shape[0] + add_num, adj_back.shape[0] + add_num), device=adj.device)
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen, :]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:, chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen, :][:, chosen]

    features_append = new_features
    labels_append = labels[chosen]
    idx_new = torch.arange(adj_back.shape[0], adj_back.shape[0] + add_num, device=idx_train.device)
    idx_train_append = idx_new

    for original_node, new_node in zip(chosen, idx_new):
        new_node_to_cluster_map[new_node.item()] = node_to_cluster_map[original_node.item()]

    features = torch.cat((features, features_append), 0)
    labels = torch.cat((labels, labels_append), 0)
    idx_train = torch.cat((idx_train, idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train, idx_new, new_node_to_cluster_map


# def src_smote(adj,features,labels,idx_train, portion=1.0, im_class_num=3):
#     c_largest = 0#labels.max().item()
#     adj_back = adj.to_dense()
#     chosen = None
#     new_features = None

#     #ipdb.set_trace()
#     avg_number = int(idx_train.shape[0]/(c_largest+1))

#     for i in range(im_class_num):
#         new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
#         if portion == 0:#refers to even distribution
#             c_portion = int(avg_number/new_chosen.shape[0])

#             portion_rest = (avg_number/new_chosen.shape[0]) - c_portion

#         else:
#             c_portion = int(portion)
#             portion_rest = portion-c_portion
            
#         for j in range(c_portion):
#             num = int(new_chosen.shape[0])
#             new_chosen = new_chosen[:num]

#             chosen_embed = features[new_chosen,:]
#             distance = squareform(pdist(chosen_embed.cpu().detach()))
#             np.fill_diagonal(distance,distance.max()+100)

#             idx_neighbor = distance.argmin(axis=-1)
            
#             interp_place = random.random()
#             embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

#             if chosen is None:
#                 chosen = new_chosen
#                 new_features = embed
#             else:
#                 chosen = torch.cat((chosen, new_chosen), 0)
#                 new_features = torch.cat((new_features, embed),0)
            
#         num = int(new_chosen.shape[0]*portion_rest)
#         new_chosen = new_chosen[:num]


        

#         chosen_embed = features[new_chosen,:]
#         distance = squareform(pdist(chosen_embed.cpu().detach()))
#         np.fill_diagonal(distance,distance.max()+100)

#         idx_neighbor = distance.argmin(axis=-1)
            
#         interp_place = random.random()
#         embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

#         if chosen is None:
#             chosen = new_chosen
#             new_features = embed
#         else:
#             chosen = torch.cat((chosen, new_chosen), 0)
#             new_features = torch.cat((new_features, embed),0)
            

#     add_num = chosen.shape[0]
#     new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
#     new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
#     new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
#     new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
#     new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

#     #ipdb.set_trace()
#     features_append = deepcopy(new_features)
#     labels_append = deepcopy(labels[chosen])
#     idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
#     idx_train_append = idx_train.new(idx_new)

#     features = torch.cat((features,features_append), 0)
#     labels = torch.cat((labels,labels_append), 0)
#     idx_train = torch.cat((idx_train,idx_train_append), 0)
#     adj = new_adj.to_sparse()

#     return adj, features, labels, idx_train

def recon_upsample(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3):
    c_largest = 0#labels.max().item()
    avg_number = int(idx_train.shape[0]/(c_largest+1))
    #ipdb.set_trace()
    adj_new = None

    for i in range(im_class_num):
        chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        num = int(chosen.shape[0]*portion)
        if portion == 0:
            c_portion = int(avg_number/chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 1

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen,:]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance,distance.max()+100)

            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()
            new_embed = embed[chosen,:] + (chosen_embed[idx_neighbor,:]-embed[chosen,:])*interp_place


            new_labels = labels.new(torch.Size((chosen.shape[0],1))).reshape(-1).fill_(c_largest-i)
            idx_new = np.arange(embed.shape[0], embed.shape[0]+chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed,new_embed), 0)
            labels = torch.cat((labels,new_labels), 0)
            idx_train = torch.cat((idx_train,idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen,:] + adj[idx_neighbor,:], min=0.0, max = 1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen,:] + adj[idx_neighbor,:], min=0.0, max = 1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[0]+add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:,:]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:,:]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train

def adj_mse_loss(adj_rec, adj_tgt, adj_mask = None):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0]**2

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss



