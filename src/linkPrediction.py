"""
In this file the link prediction method of node2vec is implemented.
"""
import copy
import itertools
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import networkx as nx

# =========================================================

def generate_neg_edges(G, num_neg_edges):
    G_comp = nx.complement(G)
    neg_edges = random.sample(G_comp.edges(), num_neg_edges)
    return neg_edges


def readEdges(path):
    """
    Function that receives a path to data set of edges and returns
    the list of nodes and edges. Assumes the network is unweighted.
    and undirected.

    :param path: path to data containing all edges
    :returns nodes, edges: list of the nodes and edges
    """
    nodes = []
    edges = []

    with open(path, 'r') as f:
        f = f.readlines()
        for edge in f:
            node1, node2 = edge.split(' ')
            nodes.append(str(int(node1)))
            nodes.append(str(int(node2)))
            edges.append((str(int(node1)), str(int(node2))))

    nodes = list(set(nodes))

    return nodes, edges


def embeddings_dict(path):
    """
    Function that opens a file containing node features and returns a dictionary of nodes as keys
    and features list as values.

    :param path: path to file containing features/embeddings of all nodes
    :returns nodes, edges: list of nodes and edges
    """
    with open(path, 'r') as file:
        file_lines = file.readlines()
    embeddings = {}
    for line in file_lines[1:]:
        line = line.split(' ')
        node = str(int(line[0]))
        feature = [float(i) for i in line[1:]]
        embeddings[node] = feature

    return embeddings


def edge_embeddings(operator, node_embeddings, edge):
    u, v = edge
    fu = node_embeddings[u]
    fv = node_embeddings[v]
    if operator == 'Average':
        return (np.array(fu) + np.array(fv)) / 2
    elif operator == 'Hadamard':
        return np.array(fu) * np.array(fv)
    elif operator == 'Weighted_L1':
        return np.abs(np.array(fu) - np.array(fv))
    elif operator == 'Weighted_L2':
        return np.power(np.array(fu) - np.array(fv), 2)
    return None


def necessary_edges(original_edges_path, node_embeddings_path, training_edges_path, test_edges_path):
    G = nx.read_edgelist(original_edges_path)  # original graph
    G_train = nx.read_edgelist(training_edges_path)  # training graph
    _, test_positive_edges = readEdges(test_edges_path)

    num_neg_train_edges = G_train.number_of_edges()
    train_negative_edges = generate_neg_edges(G, num_neg_train_edges)  # generate negative edges for training.
    train_positive_edges = list(G_train.edges())

    # for test negative edges, they should not be present in training negative edges
    G_aux = copy.deepcopy(G)
    G_aux.add_edges_from(train_negative_edges)
    num_neg_test_edges = len(test_positive_edges)
    test_negative_edges = generate_neg_edges(G_aux, num_neg_test_edges)

    node_embeddings = embeddings_dict(node_embeddings_path)

    return train_positive_edges, train_negative_edges, test_positive_edges, test_negative_edges, node_embeddings


def train_test_split(train_positive_edges, train_negative_edges, test_positive_edges,
                     test_negative_edges, node_embeddings, operator):

    # define X_train_, y_train, X_test, y_test
    X_train = []
    y_train = []
    for edge in train_positive_edges:
        edge_emb = edge_embeddings(operator, node_embeddings, edge)
        X_train.append(edge_emb)
        y_train.append(1)
    for edge in train_negative_edges:
        edge_emb = edge_embeddings(operator, node_embeddings, edge)
        X_train.append(edge_emb)
        y_train.append(0)

    X_test = []
    y_test = []
    for edge in test_positive_edges:
        edge_emb = edge_embeddings(operator, node_embeddings, edge)
        X_test.append(edge_emb)
        y_test.append(1)
    for edge in test_negative_edges:
        edge_emb = edge_embeddings(operator, node_embeddings, edge)
        X_test.append(edge_emb)
        y_test.append(0)

    # shuffle the train data
    c = list(zip(X_train, y_train))
    random.shuffle(c)
    X_train, y_train = zip(*c)
    X_train, y_train = np.array(X_train), np.array(y_train)

    # shuffle the test data
    c = list(zip(X_test, y_test))
    random.shuffle(c)
    X_test, y_test = zip(*c)
    X_test, y_test = np.array(X_test), np.array(y_test)

    print(f"Training and testing data is created!")
    return X_train, y_train, X_test, y_test


def prediction(classifier, X_train, y_train, X_test, y_test):
    """
    Function that computes the predictions of test edges, given a classifier, training data and test data
    :param classifier: a classifier
    :param train_data: list of training edges
    :param train_labels: list of labels of train edges
    :return accuracy
    """
    model = classifier.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, preds)
    print(f"----- The accuracy score is: {round(accuracy, 2)}")
    print(f"----- The auc_roc score is: {round(auc_roc, 2)}\n")

    return None


def main(original_edges_path, node_embeddings_path, training_edges_path, test_edges_path):
    """
    main function that computes the link prediction accuracy, given an operator,
    a classifier, and paths to train/test data.

    :param operators: binary operators
    :param classifier: a classifier, ex. LogisticRegression()
    :param train_embeddings_path: path to node features
    :param train_edges_path: path to training data
    :param test_edges_path: path to test data
    """
    train_positive_edges, train_negative_edges, test_positive_edges, test_negative_edges, node_embeddings = \
        necessary_edges(original_edges_path, node_embeddings_path, training_edges_path, test_edges_path)
    classifier = LogisticRegression(max_iter=1000)
    operators = ['Average', 'Hadamard', 'Weighted_L1', 'Weighted_L2']
    for operator in operators:
        X_train, y_train, X_test, y_test = \
            train_test_split(train_positive_edges, train_negative_edges, test_positive_edges,
                     test_negative_edges, node_embeddings, operator)
        print(f"Classification with binary operator {operator}: ")
        prediction(classifier, X_train, y_train, X_test, y_test)

    print(f"{40 * ('=')}\nClassification done!")


if __name__ == "__main__":
    original_edges_path = '../facebook_data/facebook_combined.txt'
    node_embeddings_path = '../facebook_data/facebook.emb'  # '../karate/karate.emb'
    training_edges_path = '../facebook_data/train_edges'  # '../karate/train.pkl'
    test_edges_path = '../facebook_data/test_edges'

    main(original_edges_path, node_embeddings_path, training_edges_path, test_edges_path)
    #nodes, edges = readEdges(original_edges_path)
    #print(nodes)
    #print(edges)
    """
    original_edges_path = '../facebook_data/facebook_combined.txt'
    node_embeddings_path = '../facebook_data/facebook.emb'  # '../karate/karate.emb'
    training_edges_path = '../facebook_data/train_edges'  # '../karate/train.pkl'
    test_edges_path = '../facebook_data/test_edges'
    """

    """
    original_edges_path = '../karate/karate.edgelist'
    node_embeddings_path = '../karate/karate.emb'  # '../karate/karate.emb'
    training_edges_path = '../karate/train_edges'  # '../karate/train.pkl'
    test_edges_path = '../karate/test_edges'
    
    original_edges_path = '../NDFRT/NDF.edgelist'
    node_embeddings_path = '../NDFRT/NDF.emb'  # '../karate/karate.emb'
    training_edges_path = '../NDFRT/train_edges'  # '../karate/train.pkl'
    test_edges_path = '../NDFRT/test_edges'
    """