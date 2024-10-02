import dgl
import networkx as nx
import pymetis

def convert_dgl_to_nx(g):
    # Convert DGL graph to NetworkX graph without node and edge attributes
    return g.to_networkx()

def partition_graph_with_metis(g, num_clusters):
    # Convert the DGL graph to a NetworkX graph
    nx_graph = convert_dgl_to_nx(g)
    
    # Get adjacency list from NetworkX graph
    adjacency_list = [list(nx_graph.neighbors(node)) for node in nx_graph.nodes()]
    
    # Perform METIS partitioning
    num_nodes = len(adjacency_list)
    _, parts = pymetis.part_graph(num_clusters, adjacency=adjacency_list)
    
    # Collect the nodes in each partition
    clusters = [[] for _ in range(num_clusters)]
    for node_id, part_id in enumerate(parts):
        clusters[part_id].append(node_id)
    
    return clusters

def metis_clusters(g, feature_matrix, train_indices, num_clusters=8, num_perm=128, threshold=0.5):


    # Partition the graph
    clusters = partition_graph_with_metis(g, num_clusters)

    # # Print the results
    # for i, cluster in enumerate(clusters):
    #     print(f"Cluster {i}: {cluster}")

    return clusters
