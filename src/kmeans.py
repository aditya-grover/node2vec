import matplotlib.pyplot as plt
import os
import networkx as nx
import numpy as np

HOMOPHILY = False

if HOMOPHILY:
    fname = 'emb/les_miserables_p_1_q_0.5.emb'
    n_clusters = 6
    plot_suffix = "homophily"
else:
    fname = 'emb/les_miserables_p_1_q_2.emb'
    n_clusters = 3
    plot_suffix = "str_eq"


os.chdir(f"c:/Users/dmitr/Documents/git/node2vec")

def save_fig(filename, h, w, dpi):
    figure = plt.gcf()
    figure.set_size_inches(w, h)
    plt.savefig(filename, dpi=dpi)
    
G = nx.generators.social.les_miserables_graph
G().number_of_nodes()
G().number_of_edges()
list(G().edges)
G()["Javert"]

d = dict(G().degree)
# node_size = [v * 100 for v in d.values()]
# Discretize node degrees
step = int(max(d.values()) / 5)
node_size = [int(e / step + 1) * 200 + 150 for e in d.values()]

subax1 = plt.subplot()

nx.draw(G(), with_labels=True, node_size=node_size, font_weight='bold')
save_fig("images/les_miserables_draw.png", h = 10, w = 20, dpi = 200)
plt.show()

# nx.draw_circular(G(), with_labels=True, node_size=node_size, font_weight='bold')
# save_fig("images/les_miserables_draw_circular.png", h = 10, w = 20, dpi = 200)
# plt.show()


G_int = nx.convert_node_labels_to_integers(G(), first_label=0)

nx.readwrite.edgelist.write_edgelist(G = G_int, path = "graph/les_miserables.edgelist", data = False)

# RUN THE NODE2VEC SCRIPT

"""
python src/main.py --input graph/les_miserables.edgelist --output emb/les_miserables_p_1_q_05.emb --dimensions 16 --p 1 --q 0.5
"""

with open(fname) as f:
    # ignore the first line.
    emb = f.read().splitlines()[1:]

# Split with whitespace.
emb = [e.split() for e in emb]
# Convert embeddings to float.
emb = {e[0]: [float(ee) for ee in e[1:]] for e in emb}

emb_lst = list(emb.values())


# Clustering

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb_lst)
kmeans.labels_



# Plot the original graph with colours according to the clusters.

node_names = list(G().nodes())
node_keys = list(emb.keys())
node_clusters = kmeans.labels_

from matplotlib import cm
cmap = cm.get_cmap('Set1', 6)

color_map = []
for i in range(G().number_of_nodes()):
    cluster = node_clusters[i]
    color_map.append(cmap(cluster))
 
nx.draw(G(), node_color=color_map, with_labels=True, node_size=node_size, font_weight='bold')
save_fig(f"images/les_miserables_draw_kmeans_{plot_suffix}.png", h = 10, w = 20, dpi = 200)
plt.show()

# nx.draw_circular(G(), node_color=color_map, with_labels=True, node_size=node_size, font_weight='bold')
# save_fig("images/les_miserables_draw_circular_kmeans.png", h = 10, w = 20, dpi = 200)
# plt.show()


