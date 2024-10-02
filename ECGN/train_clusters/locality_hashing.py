import dgl
import networkx as nx
import pymetis
from datasketch import MinHash, MinHashLSH
import torch

def graph_to_adjacency_list(nx_graph):
    adjacency_list = []
    for node in sorted(nx_graph.nodes()):
        neighbors = list(nx_graph.neighbors(node))
        adjacency_list.append(neighbors)
    return adjacency_list



def locality_hashing_clusters(g, feature_matrix, train_indices, num_clusters=8, num_perm=128, threshold=0.5):
    # Ensure feature_matrix is in float32 format
    feature_matrix = feature_matrix.to(torch.float32)

    # Initialize LSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Create MinHash objects and add them to LSH
    minhashes = []
    for i in range(feature_matrix.shape[0]):
        minhash = MinHash(num_perm=num_perm)
        # Convert the tensor row to a list of bytes and update MinHash
        feature_vector = feature_matrix[i].cpu().numpy()
        for bit in feature_vector:
            minhash.update(bit.tobytes())
        minhashes.append(minhash)
        lsh.insert(f'node_{i}', minhash)

    # Function to get clusters and return as a nested list
    def get_clusters_as_nested_list(lsh, num_clusters):
        clusters = {}
        cluster_map = {}
        cluster_id = 0

        for i in range(feature_matrix.shape[0]):
            node_id = f'node_{i}'
            neighbors = lsh.query(minhashes[i])

            # Check if any neighbors are already assigned to a cluster
            assigned_cluster = None
            for neighbor in neighbors:
                if neighbor in cluster_map:
                    assigned_cluster = cluster_map[neighbor]
                    break

            # Assign a new cluster if no assignment was found
            if assigned_cluster is None:
                assigned_cluster = cluster_id
                cluster_id = (cluster_id + 1) % num_clusters

            cluster_map[node_id] = assigned_cluster

            if assigned_cluster not in clusters:
                clusters[assigned_cluster] = []
            clusters[assigned_cluster].append(i)

        # Convert to a nested list
        nested_list = [clusters.get(i, []) for i in range(num_clusters)]
        return nested_list

    # Get nested list of clusters
    clusters_nested_list = get_clusters_as_nested_list(lsh, num_clusters)

    # Merge small clusters without train indices
    min_cluster_size = 15
    train_indices_set = set(train_indices)
    new_clusters = []
    small_clusters = []

    for cluster in clusters_nested_list:
        cluster_set = set(cluster)
        if len(cluster_set) < min_cluster_size and not cluster_set.intersection(train_indices_set):
            small_clusters.append(cluster_set)
        else:
            new_clusters.append(cluster_set)

    # Merge all small clusters into one
    if small_clusters:
        merged_cluster = set().union(*small_clusters)
        new_clusters.append(list(merged_cluster))

    # Print the results
    total_nodes = sum(len(cluster) for cluster in new_clusters)
    print(f"Total nodes: {total_nodes}, Feature matrix shape: {feature_matrix.shape}")
    print(f"Number of clusters: {len(new_clusters)}")

    return new_clusters
