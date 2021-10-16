"""
In this file the link prediction method of node2vec is implemented.
"""
import pickle
import networkx
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score



def openFile(data):
    """
    Function that receives a data set together with the number of nodes and edges
    of edges and returns an adjacency matrix. Assume edges in data are unweighted
    and undirected.
    """
    nodes = []
    edges = []

    with open(data, 'r') as f:
        f = f.readlines()
        for edge in f:
            node1, node2 = edge.split(' ')
            nodes.append(int(node1))
            nodes.append(int(node2))
            edges.append((int(node1), int(node2)))

    nodes = list(set(nodes))

    return nodes, edges


def openFeatures(path):
    with open(path, 'r') as file:
        file_lines = file.readlines()
    nodes = []
    features = []
    for line in file_lines[1:]:
        line = line.split(' ')
        node = int(line[0])
        nodes.append(node)
        feature = [float(i) for i in line[1:]]
        features.append(feature)

    return nodes, features


def openPkl(path):
    lst = []
    with open(path, 'rb') as file:
        while True:
            try:
                l = pickle.load(file)
                lst.append(l)
            except EOFError:
                break
    return lst


def createGraph(nodes, edges, name):
    """
    From an adjacency matrix create and return a graph.
    """
    G = networkx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    G_complement = networkx.complement(G)

    negative_edges = list(G_complement.edges())

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set(title=f'Graph for the {name} dataset')
    networkx.draw(G, with_labels=True)
    fig.savefig(f'{name}.png')
    plt.show()

    return G, edges, negative_edges


def pickleEdges(path, edges_list):
    with open(path, 'wb') as file:
        for lst in edges_list:
            pickle.dump(lst, file)
    return None


def node2vec_format(path, edges):
    with open(path, 'w') as file:
        for edge in edges:
            u, v = edge
            file.write(f"{u} {v}\n")
    return None


def removeEdges(G, edges, negative_edges, name, path):
    """
    remove 50% of the existing_edges from the graph G, s.t. the residual graph is connected.
    and add an equal number of non_existing_edges to the list, to be used as test set.
    """

    n = G.number_of_edges()//2
    test_edges = []
    sampled_real_edges = []
    test_labels = []

    # add a random number of non-edges to test set
    non_edges_index = np.random.choice(range(len(negative_edges)), size=n, replace=False)
    sampled_negative_edges = [negative_edges[i] for i in non_edges_index]
    test_edges.extend(sampled_negative_edges)
    test_labels.extend([0 for _ in range(n)])  # labels for the non-edges

    # remove real edges from the graph while keeping it connected, and add the edge to test_edges
    while n > 0:
        sample_edge = np.random.choice(range(len(edges)))  # first select an edge at random
        sample_edge = edges[sample_edge]

        u, v = sample_edge
        G.remove_edge(u, v)

        # check that the graph is connected
        if networkx.algorithms.components.number_connected_components(G) == 1:
            # if yes, then remove the sampled edge from the list of edges, and decrement n
            test_edges.append(sample_edge)
            sampled_real_edges.append(sample_edge)
            test_labels.append(1)
            edges.remove(sample_edge)
            n -= 1
        else:
            u, v = sample_edge
            G.add_edge(u, v)  # we have to add back the sampled edge

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set(title=f'{name} graph after edge removal')
    networkx.draw(G, with_labels=True)
    fig.savefig(f'{name}_remove.png')
    plt.show()

    # constructing training real edges and negative edges, as well as their labels
    train_edges = edges
    node2vec_format(f'{path}/train_feature.edgelist', train_edges)

    train_edges_complement = list(networkx.complement(G).edges())
    sampled_complement_edges = np.random.choice(range(len(train_edges_complement)), size=len(train_edges), replace=False)
    sampled_complement_edges = [train_edges_complement[i] for i in sampled_complement_edges]

    train_edges.extend(sampled_complement_edges)
    train_labels = [1 for _ in range(len(train_edges))]
    train_labels.extend([0 for _ in range(len(sampled_complement_edges))])

    pickleEdges(f"{path}/train.pkl", [train_edges, train_labels])
    pickleEdges(f"{path}/test.pkl", [test_edges, test_labels])

    return train_edges, test_edges, test_labels, sampled_real_edges, sampled_negative_edges


def EdgeFeatures(oper, nodes, node_features, train_edges):
    edge_features = []
    for edge in train_edges:
        u, v = edge  # the nodes in the edge
        ui, vi = nodes.index(u), nodes.index(v)  # their indices in the nodes
        fu, fv = node_features[ui], node_features[vi]
        ef = 0  # edge feature
        if oper == 'Average':
            print(np.array(fu))
            ef = (np.array(fu) + np.array(fv)) / 2
        elif oper == 'Hadamard':
            ef = np.array(fu) * np.array(fv)
        elif oper == 'Weighted_L1':
            ef = np.abs(np.array(fu) - np.array(fv))
        elif oper == 'Weighted_L2':
            ef = np.pow(np.array(fu) - np.array(fv), 2)
        edge_features.append(ef)

    return edge_features


def train_classifier(train_embeddings_path, train_edges_path):
    nodes, train_feature_edges = openFeatures(train_embeddings_path)
    train_edges, train_labels = openPkl(train_edges_path)
    oper = 'Avgerage'  # one of four operators [Avgerage, Hadamard, Weighted_L1, Weighted_L2]
    edge_features = EdgeFeatures(oper=oper, nodes=nodes, node_features=train_feature_edges, train_edges=train_edges)
    print(edge_features)
    df = pd.DataFrame()

    # once edge_features is ready:
    # create a df of edge features for each edge
    #train_df = pd.DataFrame(edge_features, index=train_edges, columns=[f"f{i}" for i in range(len(train_edges[0]))])
    #
    # # choose a classifier
    # classifier = LogisticRegression()
    # scores = cross_val_score(classifier, train_df, train_labels, cv=5)
    # print(f"scores: {scores}")

    # train it using 5 fold cross-val


    # open test features and labels
    # compute their embeddings
    # create df of them
    # test classifier on them


if __name__ == "__main__":
    # facebook = '../facebook_data/facebook_combined.txt'
    toy = '../toyExample/toy.edgelist'
    # karate = '../karate/karate.edgelist'
    # # num_nodes = 4039
    # # num_edges = 88234
    #nodes, edges = openFile(data=karate)
    # #
    #g, edges, negative_edges = createGraph(nodes=nodes, edges=edges, name='karate')
    # #
    # train_edges, test_edges, labels, sampled_real_edges, sampled_negative_edges = \
    #     removeEdges(g, edges, negative_edges, 'karate', '../karate')

    train_classifier(train_embeddings_path='../karate/karate.emd', train_edges_path='../karate/train.pkl')
    #nodes, edges = openFeatures('../karate/karate.emd')
    #print(f"Nodes: {nodes