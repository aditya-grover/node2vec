import matplotlib.pyplot as plt
import os
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import cm


os.chdir(f"c:/Users/dmitr/Documents/git/node2vec")

SWITCH = "str_eq" # homophily, str_eq, struc2vec
HOMOPHILY = False
EDGELIST_FNAME = "graph/les_miserables_int.edgelist"
D = 16
P = 1
K = 10 # window_size = context size k, default = 10.
L = 80 # walk_length = l, default = 80.

def get_emb_fname_for_node2vec():
    params = f"d_{D}_l_{L}_k_{K}_p_{P}_q_{Q}"
    emb_fname = f"emb/les_miserables_{params}.emb"
    return params, emb_fname

def save_fig(filename, h, w, dpi):
    figure = plt.gcf()
    figure.set_size_inches(w, h)
    plt.savefig(filename, dpi=dpi)

def do_elbow_method():
    # Source: https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion
    sse = {}
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(emb_lst)
        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    save_fig("images/elbow.png", h = 5, w = 7, dpi = 200)
    plt.show()
    
    
if SWITCH == "homophily":
    print("Search for homophily with node2vec...")
    Q = 0.5
    n_clusters = 6
    plot_suffix = "node2vec_homophily"
    params, emb_fname = get_emb_fname_for_node2vec()
    
elif SWITCH == "str_eq":
    print("Search for structural equivalence with node2vec...")
    Q = 2
    n_clusters = 3
    plot_suffix = "node2vec_str_eq"
    params, emb_fname = get_emb_fname_for_node2vec()
    
elif SWITCH == "struc2vec":
    print("Using for structural equivalence with struc2vec...")
    n_clusters = 4
    emb_fname = "../BioNEV/embeddings/les_miserables_struc2vec.emb"
    plot_suffix = "str_eq"
    params = "struc2vec"
    
else:
    raise Exception('Switch should be "homophily", "str_eq" or "struc2vec".')



# Draw original graph -------------------------------------
G = nx.generators.social.les_miserables_graph()

# G.number_of_nodes()
# G.number_of_edges()
# list(G.edges)
# G["Javert"]

d = dict(G.degree)
step = int(max(d.values()) / 5) # Discretize node degrees into 5 groups (~ as in Figure 3).
node_size = [int(e / step + 1) * 200 + 150 for e in d.values()]

subax1 = plt.subplot()
nx.draw(G, with_labels=True, node_size=node_size)
save_fig("images/les_miserables_draw.png", h = 10, w = 20, dpi = 200)
plt.show()


# Export edgelist -------------------------------------

G_int = nx.convert_node_labels_to_integers(G, first_label = 0)
nx.readwrite.edgelist.write_edgelist(G = G_int, path = EDGELIST_FNAME, data = False)


# Execute node2vec -------------------------------------
if SWITCH == "struc2vec":
    # Install BioNEV package according to the instructions in the README: 
        # https://github.com/xiangyue9607/BioNEV.
    # Run 
        # cd ../BioNEV
        # bionev --input ../node2vec/graph/les_miserables_int.edgelist --output ./embeddings/les_miserables_struc2vec.emb --method struc2vec --task link-prediction --walk-length 80 --window-size 10 --dimensions 16
    pass
else:
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

node_names = list(G.nodes()) # Used to replace integers with character names.

with open(emb_fname) as f:
    # Ignore the first line (# of nodes, # of dimensions).
    emb = f.read().splitlines()[1:]

emb = [e.split() for e in emb] # Split with whitespace.
emb = {node_names[int(e[0])]: [float(ee) for ee in e[1:]] for e in emb} # Convert embeddings to float.
emb_lst = list(emb.values())


# Clustering -------------------------------------

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb_lst)
kmeans.labels_

if SWITCH == "struc2vec":
    do_elbow_method()


# Plot the original graph with colours according to the clusters -------------------------------------

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

