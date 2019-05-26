import numpy as np
import csv


def get_as_numpy_array(file_path):
    with open(file_path) as file:
        features = csv.reader(file, delimiter=' ')
        header = next(features, None)
        feature_array = np.zeros((int(header[0]), int(header[1])))
        for feature in features:
            feature_array[int(feature[0]) - 1] = feature[1:]
        return np.array(feature_array, dtype=float)


def adjacency_matrix_to_edgelist(matrix):
    edgelist = []
    for i, row in enumerate(matrix):
        for j, el in enumerate(row):
            if el != 0:
                edgelist.append((i+1, j+1))
    return edgelist


def reduce_if_exceeds(tuple, treshold=16, reduction=1):
    a, b = tuple
    if a > treshold:
        a -= reduction
    if b > treshold:
        b -= reduction
    return a, b


with open('../graph/lesmis.matrix') as file:
    matrix = csv.reader(file, delimiter=' ')
    np_matrix = np.array(list(matrix), dtype=int)
    edgelist = adjacency_matrix_to_edgelist(np_matrix)

# edgelist = [reduce_if_exceeds(t) for t in edgelist]
# edgelist = [reduce_if_exceeds(t, treshold=45) for t in edgelist]
with open('../graph/lesmis.edgelist', 'w') as file:
    csv_writer = csv.writer(file, delimiter=' ')
    csv_writer.writerows(edgelist)
