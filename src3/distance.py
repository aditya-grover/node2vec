import csv
import numpy as np
import scipy
import math

from sklearn import mixture
from sklearn import cluster


def cos_distance(a, b):
    return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calc_principal_angels(A, B):
    _, s, _ = np.linalg.svd(A.T @ B)
    return s


def principal_angle_distance(embeddings):
    res = [scipy.linalg.orth(el[0]) for el in embeddings]

    for el1 in res:
        for el2 in res:
            principal_angels_cos = calc_principal_angels(el1, el2)
            principal_angels = [math.degrees(math.acos(min(el, 1))) for el in principal_angels_cos]
            print(principal_angels_cos)
            print(principal_angels)
            print()


def get_as_numpy_array(file_path):
    with open(file_path) as file:
        features = csv.reader(file, delimiter=' ')
        header = next(features, None)
        feature_array = np.zeros((int(header[0]), int(header[1])))
        for feature in features:
            feature_array[int(feature[0]) - 1] = feature[1:]
        return np.array(feature_array, dtype=float)


def calc_matrix_norm(matrix_list):
    for i in range(len(matrix_list)):
        for j in range(len(matrix_list)):
            if i != j:
                print("{} with {} ".format(i + 1, j + 1) + str(np.linalg.norm(matrix_list[i] - matrix_list[j])))


def get_matrixs(pattern, count):
    return [get_as_numpy_array(pattern.format(i + 1)) for i in range(count)]


def get_cluster_sim(c1, c2):
    s = 0
    for a, b in zip(c1, c2):
        if a != b:
            s += 1
    return s


def map_clusters(c1, c2, k):
    cluster_mapping = {}
    for i in range(k):
        cluster_mapping[i] = {}
    for a, b in zip(c1, c2):
        cluster_mapping[a][b] = cluster_mapping[a].get(b, 0) + 1
    return {entry[0]: max(entry[1].items(), key=lambda el: el[1])[0] for entry in cluster_mapping.items()}



def calc_cluster_distance(c1, c2, k, method):
    # gmmA = mixture.BayesianGaussianMixture(n_components=k, covariance_type='full').fit(A)
    # gmmB = mixture.BayesianGaussianMixture(n_components=k, covariance_type='full').fit(B)
    # kmeansA = cluster.KMeans(n_clusters=k).fit(A)
    # kmeansB = cluster.KMeans(n_clusters=k).fit(B)
    # predictA = gmmA.predict(A)
    # predictA = kmeansA.labels_
    # predictB = gmmB.predict(B)
    # predictB = kmeansB.labels_
    mapping = map_clusters(c1, c2, k)
    c1_m = [mapping[el] for el in c1]
    s = get_cluster_sim(c1_m, c2)
    print(list(c1))
    print(list(c2))
    print(c1_m)
    print(f"Different assignments for {method} with {k} clusters: {s}, which gives {(s / len(c1)) * 100:10.2f}%")

def get_gmm_clusters(X, k):
    gmm = mixture.BayesianGaussianMixture(n_components=k, covariance_type='full').fit(X)
    return gmm.predict(X)

def get_km_clusters(X, k):
    return cluster.KMeans(n_clusters=k, random_state=0).fit(X).labels_

def cluster_distance(embeddings, k):
    embeddings = [sorted(el, key=lambda e: e[1]) for el in embeddings]
    clear_embeddings = []
    for emb in embeddings:
        clear_emb = []
        for el in emb:
            clear_emb.append(el[0])
        clear_embeddings.append(clear_emb)
    for i, el1 in enumerate(clear_embeddings):
        for j, el2 in enumerate(clear_embeddings):
            print(f"{i}/{j}")
            calc_cluster_distance(get_gmm_clusters(el1, k), get_gmm_clusters(el2, k), k, "BGMM")
            calc_cluster_distance(get_km_clusters(el1, k), get_km_clusters(el2, k), k, "KMeans")
            print("---------------")
# calc_matrix_norm(get_matrixs('../emb/lesmis{}.emb', 4))

# for i in range(len(f1)):
#     print cos_distance(f1[i], f2[i])
