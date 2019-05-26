import itertools

import numpy as np
from scipy import linalg
import matplotlib as mpl
import csv
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn import cluster
from utils import get_as_numpy_array

G = nx.Graph()

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    # plt.xlim(-3., 2.)
    # plt.ylim(-2., 3.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


with open('../graph/lesmis.edgelist') as graph_file:
    graph_csv = csv.reader(graph_file, delimiter=' ')
    for row in graph_csv:
        G.add_edge(int(row[0]), int(row[1]))

d = nx.degree(G)

X = get_as_numpy_array('../emb/lesmis.emb')

gmm = mixture.BayesianGaussianMixture(n_components=6, covariance_type='full').fit(X)
kmeans = cluster.KMeans(n_clusters=3, random_state=0).fit(X)
print kmeans.labels_
# prediction = gmm.predict(X)
prediction = kmeans.labels_
# plot_results(X, prediction, gmm.means_, gmm.covariances_, 1,
#              'Bayesian Gaussian Mixture with a Dirichlet process prior')

nx.draw(G, node_size=[v * 10 for v in d.values()],
        node_color=prediction)
plt.show()
