import matplotlib.pyplot as plt
import os
import networkx as nx
from sklearn.cluster import KMeans
from matplotlib import cm

SWITCH = "homophily" # homophily, str_eq, struc2vec

DATA_NAME = "les_miserables" # les_miserables, TerroristRel
# TerroristRel: https://networkrepository.com/TerroristRel.php

args = {"edgelist_fname": f"graph/{DATA_NAME}/{DATA_NAME}.edgelist",
        "edgelist_delim": ",",
        "D": 16, 
        "P": 1,
        "K": 10, # window_size = context size k, default = 10.
        "L": 80 # walk_length = l, default = 80.
        }

def get_data():
    """Get data as a networkx graph."""
    
    if DATA_NAME == "les_miserables":
        return nx.generators.social.les_miserables_graph()
    else:
        return nx.readwrite.edgelist.read_edgelist(f"graph/{DATA_NAME}/{DATA_NAME}.edges", 
                                                   delimiter = args["edgelist_delim"])

def set_other_parameters():
    """Set q and the number of clusters as in the original paper,
    define some string variables to be used in filenames."""

    if SWITCH == "homophily":
        print("Search for homophily with node2vec...")
        args["Q"] = 0.5
        args["n_clusters"] = 6
        args["plot_suffix"] = "node2vec_homophily"
        get_emb_fname_for_node2vec()
        
    elif SWITCH == "str_eq":
        print("Search for structural equivalence with node2vec...")
        args["Q"] = 2
        args["n_clusters"] = 3
        args["plot_suffix"] = "node2vec_str_eq"
        get_emb_fname_for_node2vec()
        
    elif SWITCH == "struc2vec":
        print("Using for structural equivalence with struc2vec...")
        args["n_clusters"] = 4
        args["emb_fname"] = f"emb/{DATA_NAME}/{DATA_NAME}_struc2vec.emb"
        args["plot_suffix"] = "str_eq"
        args["params"] = "struc2vec"
        
    else:
        raise Exception('Switch should be "homophily", "str_eq" or "struc2vec".')

def get_emb_fname_for_node2vec():
    """Returns two strings:
    params - concatenated parameters to be used in the plot files.
    emb_fname - filename for the embeddings file.
    """
    
    args["params"] = f"d_{args['D']}_l_{args['L']}_k_{args['K']}_p_{args['P']}_q_{args['Q']}"
    args["emb_fname"] = f"emb/{DATA_NAME}/{DATA_NAME}_{args['params']}.emb"

def save_fig(filename, h, w, dpi):
    """Save the plot with specified height, width and dpi parameters."""
    
    figure = plt.gcf()
    figure.set_size_inches(w, h)
    plt.savefig(filename, dpi=dpi)
   
def export_edgelist(G):
    """Export G as an edgelist with integer IDs."""
    
    if DATA_NAME == "les_miserables":
        # Because we took it from networkx package.
        G_int = nx.convert_node_labels_to_integers(G, first_label = 0)
        nx.readwrite.edgelist.write_edgelist(G = G_int, path = args["edgelist_fname"], data = False)

def create_embeddings():
    """Execute node2vec or struc2vec to create embeddings."""

    if SWITCH == "struc2vec":
        # Install BioNEV package according to the instructions in the README: 
            # https://github.com/xiangyue9607/BioNEV.
        # Run 
            # cd ../BioNEV
            # bionev --input ../node2vec/graph/les_miserables.edgelist --output ../node2vec/emb/les_miserables_struc2vec.emb --method struc2vec --task link-prediction --walk-length 80 --window-size 10 --dimensions 16
        pass
    else:
        cmd = "python src/main.py " + \
            "--input " + args["edgelist_fname"] + \
            " --output " + args["emb_fname"] + \
            " --dimensions " + str(args["D"]) + \
            " --walk-length " + str(args["L"]) + \
            " --window-size " + str(args["K"]) + \
            " --p " + str(args["P"]) + \
            " --q " + str(args["Q"])
    
        print(f"Command: {cmd}")
        os.system(cmd)

def read_embeddings(G):
    """Read embeddings from an external file."""
    
    node_names = list(G.nodes()) # Used to replace integers with character names.
    
    with open(args["emb_fname"]) as f:
        # Ignore the first line (# of nodes, # of dimensions).
        emb = f.read().splitlines()[1:]
    
    emb = [e.split() for e in emb] # Split with whitespace.
    emb = {node_names[int(e[0])]: [float(ee) for ee in e[1:]] for e in emb} # Convert embeddings to float.
    emb_lst = list(emb.values())
    
    return emb, emb_lst, node_names

def do_clustering(emb_lst):
    """Do the kmeans clustering with embeddings as features."""

    kmeans = KMeans(n_clusters=args["n_clusters"], random_state=0).fit(emb_lst)
    kmeans.labels_
        
    return kmeans

def plot_elbow_method(emb_lst):
    """Plot and save an elbow method for kmeans."""
    
    # Source: https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion
    sse = {}
    for k in range(1, 30):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(emb_lst)
        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    save_fig(f"images/{DATA_NAME}/elbow.png", h = 5, w = 7, dpi = 200)
    plt.show()

def plot_original_network(G):
    """Plot the network and export into a file"""
    
    subax1 = plt.subplot()
    nx.draw(G, with_labels=True, alpha = 0.7)
    save_fig(f"images/{DATA_NAME}/{DATA_NAME}_draw.png", h = 10, w = 20, dpi = 200)
    plt.show()

def plot_network_with_clusters(emb, kmeans, node_names):
    """Plot the original graph with colours according to the clusters."""

    node_keys = list(emb.keys())
    node_clusters = kmeans.labels_
    
    
    cmap = cm.get_cmap('Set1', args["n_clusters"])
    color_map = []
    for node_name in node_names:
        name_index = node_keys.index(node_name)
        cluster = node_clusters[name_index]
        color_map.append(cmap(cluster))
     
    nx.draw(G, node_color=color_map,
            with_labels = True, # CHANGE HERE TO ADD LABELS!
            alpha = 0.7)
    save_fig(f"images/{DATA_NAME}/{DATA_NAME}_{args['params']}_kmeans_{args['n_clusters']}_clusters_{args['plot_suffix']}.png", 
             h = 10, 
             w = 20, 
             dpi = 200)
    plt.show()


if __name__ == "__main__":
    set_other_parameters()
    G = get_data()
    plot_original_network(G)
    export_edgelist(G)
    create_embeddings()
    emb, emb_lst, node_names = read_embeddings(G)
    kmeans = do_clustering(emb_lst)
    plot_elbow_method(emb_lst)
    plot_network_with_clusters(emb, kmeans, node_names)
