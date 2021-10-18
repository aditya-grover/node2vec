"""
In this file the necessary processing of network data is implemented.
A data set containing edges for unweighted graph can be called data.edgelist living in a directory called data.
The training data will be saved into a file at data/train_edges
The testing edges will be saved into a file at data/test_edges
For creating node embeddings, run the main function of node2vec,
where input should be data/train_edges and path for the embeddings should be data/data.emb
"""

import copy
import pickle
import networkx
import numpy as np
import matplotlib.pyplot as plt
import random


# =========================================================
def node2vec_format(path, edges):
    """
    function that stores edges in a file in the format expected by node2vec algorithms.

    :param path: path to the file.
    :param edges:  list of edges
    """
    with open(path, 'w') as file:
        for edge in edges:
            u, v = edge
            file.write(f"{u} {v}\n")
    print(f"Test edges stored in node2vec expected format, at: {path}")
    return None


def drawGraph(G, figname):
    """
    Given a graph, make a drawing and save figure with given name.

    :param G: graph
    :param figname: name of figure to be saved as.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set(title=f'Graph for the {figname} dataset')
    networkx.draw(G, with_labels=False)
    fig.savefig(f'{figname}')
    print(f"Graph created!\n")


def createGraph(input_edge_list, figname):
    """
    From list of nodes and edges create and return a graph.
    Plus saves a drawing of the graph.

    :param nodes: list of nodes
    :param edges: list of edges
    :param figname: figure name
    """
    print(f"Graph creation started: ")
    G = networkx.read_edgelist(input_edge_list)
    print(f"----- Original graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    # only consider the largest connected component
    G = G.subgraph(max(networkx.connected_components(G), key=len)).copy()
    print(f"----- The largest component subgraph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.\n")
    #drawGraph(G=G, figname=figname)
    return G


def split_train_test_graph(input_path, output_train_path, output_test_path, figname, testing_data_ratio):
    G = createGraph(input_path, figname)  # create the original graph
    numEdges = G.number_of_edges()

    num_pos_edges = int(testing_data_ratio*numEdges)
    test_positive_edges = random.sample(G.edges(), num_pos_edges)

    # removal of positive test edges:
    print(f"Edge removal started: ")
    G_train = G
    i = 0
    for edge in test_positive_edges:
        if i % 1000 == 0:
            print(f"----- {num_pos_edges-i} more edges left for removal")
        u, v = edge
        # if removing the edge does not create isolated nodes, then remove it
        if G_train.degree(u) > 1 and G_train.degree(v) > 1:
            G_train.remove_edge(u, v)
            i += 1

    #drawGraph(G_train, f'{figname}_removal')  # draw resulting graph
    # write out list of train edges that can be used for node embedding and classifier training
    networkx.write_edgelist(G_train, output_train_path, data=False)
    node2vec_format(output_test_path, test_positive_edges)  # store test positive edges
    print(f"----- Edge removal done. Training edges stored at {output_train_path}")
    return None


def main(input_path, output_train_path, output_test_path, figname, testing_data_ratio=0.2):
    split_train_test_graph(input_path, output_train_path, output_test_path, figname, testing_data_ratio)


if __name__ == "__main__":
    input_path = '../NDFRT/NDF.edgelist'
    figname = 'NDFRT'
    output_train_path = '../NDFRT/train_edges'
    output_test_path = '../NDFRT/test_edges'
    testing_data_ratio = 0.2

    main(input_path, output_train_path, output_test_path, figname, testing_data_ratio)

    """    
    input_path = '../facebook_data/facebook_combined.txt'  # '../karate/karate.edgelist'
    figname = 'facebook'  # 'karate'
    output_train_path = '../facebook_data/train_edges'  # '../karate'
    output_test_path = '../facebook_data/test_edges'  # '../karate'
    testing_data_ratio = 0.2
    """

    """
    input_path = '../NDFRT/NDF.edgelist'
    figname =  'NDFRT'
    output_train_path = '../NDFRT/train_edges'
    output_test_path = '../NDFRT/test_edges'
    testing_data_ratio = 0.2
    
    input_path = '../karate/karate.edgelist'
    figname =  'karate'
    output_train_path = '../karate/train_edges'
    output_test_path = '../karate/test_edges'
    testing_data_ratio = 0.2
    """