import matplotlib.pyplot as plt
import os
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import cm


os.chdir(f"c:/Users/dmitr/Documents/git/node2vec")

HOMOPHILY = False
EDGELIST_FNAME = "graph/les_miserables.edgelist"
D = 16
P = 1
K = 10 # window_size = context size k, default = 10.
L = 80 # walk_length = l, default = 80.

if HOMOPHILY:
    print("Search for homophily...")
    Q = 0.5
    n_clusters = 6
    plot_suffix = "homophily"
else:
    print("Search for structural equivalence...")
    Q = 2
    n_clusters = 3
    plot_suffix = "str_eq"

params = f"d_{D}_l_{L}_k_{K}_p_{P}_q_{Q}"
emb_fname = f"emb/les_miserables_{params}.emb"


def save_fig(filename, h, w, dpi):
    figure = plt.gcf()
    figure.set_size_inches(w, h)
    plt.savefig(filename, dpi=dpi)


# Draw original graph -------------------------------------
G = nx.generators.social.les_miserables_graph()

# G.number_of_nodes()
# G.number_of_edges()
# list(G.edges)
# G["Javert"]

d = dict(G.degree)
# Discretize node degrees into 5 groups (~ as in Figure 3).
step = int(max(d.values()) / 5)
node_size = [int(e / step + 1) * 200 + 150 for e in d.values()]

subax1 = plt.subplot()
nx.draw(G, with_labels=True, node_size=node_size)
save_fig("images/les_miserables_draw.png", h = 10, w = 20, dpi = 200)
plt.show()


# Export edgelist -------------------------------------

# G_int = nx.convert_node_labels_to_integers(G, first_label=0)
# nx.readwrite.edgelist.write_edgelist(G = G_int, path = "graph/les_miserables.edgelist", data = False)


# Execute node2vec -------------------------------------
cmd = "python src/main.py " + \
    "--input " + EDGELIST_FNAME + \
    " --output " + emb_fname + \
    " --dimensions " + str(D) + \
    " --walk-length " + str(L) + \
    " --window-size " + str(K) + \
    " --p " + str(P) + \
    " --q " + str(Q)

print(f"Command: {cmd}")
os.system(cmd)


# Read embeddings -------------------------------------

with open(emb_fname) as f:
    # Ignore the first line (# of nodes, # of dimensions).
    emb = f.read().splitlines()[1:]

emb = [e.split() for e in emb] # Split with whitespace.
emb = {e[0]: [float(ee) for ee in e[1:]] for e in emb} # Convert embeddings to float.
emb_lst = list(emb.values())


# Clustering -------------------------------------

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb_lst)
kmeans.labels_


# Plot the original graph with colours according to the clusters -------------------------------------


node_names = list(G.nodes()) # <-- problem?
node_keys = list(emb.keys())
node_clusters = kmeans.labels_

cmap = cm.get_cmap('Set1', n_clusters)
color_map = []
for node_name in node_names:
    name_index = node_keys.index(node_name)
    # print(f"{node_name} has index {name_index} in the edgelist.")
    cluster = node_clusters[name_index] # <-- problem?
    color_map.append(cmap(cluster)) # <-- problem?
 
nx.draw(G, node_color=color_map, with_labels=True, node_size=node_size)
save_fig(f"images/les_miserables_{params}_draw_kmeans_{plot_suffix}.png", h = 10, w = 20, dpi = 200)
plt.show()

