import csv
import numpy as np


def cos_distance(a, b):
    return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_as_numpy_array(file_path):
    with open(file_path) as file:
        features = csv.reader(file, delimiter=' ')
        header = next(features, None)
        feature_array = np.zeros((int(header[0]), int(header[1])))
        for feature in features:
            feature_array[int(feature[0]) - 1] = feature[1:]
        return np.array(feature_array, dtype=float)


def calc_matrix_norm(matrix_list):
    for i in xrange(len(matrix_list)):
        for j in xrange(i + 1, len(matrix_list)):
            if i != j:
                print "{} with {} ".format(i + 1, j + 1) + str(np.linalg.norm(matrix_list[i] - matrix_list[j]))


def get_matrixs(pattern, count):
    return [get_as_numpy_array(pattern.format(i + 1)) for i in xrange(count)]


calc_matrix_norm(get_matrixs('../emb/lesmis{}.emb', 4))

# for i in range(len(f1)):
#     print cos_distance(f1[i], f2[i])
