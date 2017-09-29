'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

from utils import convert_embed_to_np
import argparse
import os
import networkx as nx
import node2vec
import cPickle as pkl
import numpy as np
from gensim.models import Word2Vec

c = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input_name', nargs='?', default='flickr',
                        help='Input graph path. karate or cora or blog or flickr.')

    parser.add_argument('--input_type', nargs='?', default='txt',
                        help='Input graph path. edgelist or graph or npy or txt.')

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=1,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.input_type == 'edgelist':
        input = '%s/../graph/%s.edgelist' % (c, args.input_name)
        if args.weighted:
            G = nx.read_edgelist(input, nodetype=int, data=(('weight', float),),
                                 create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
            print '@@@', G
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1
    elif args.input_type == 'graph':
        assert (not args.weighted)
        input = '%s/../graph/ind.%s.graph' % (c, args.input_name)
        with open(input) as f:
            G = nx.from_dict_of_lists(pkl.load(f))
        print G
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    elif args.input_type == 'npy':
        assert (not args.weighted)
        input = '%s/../graph/%s_adj.npy' % (c, args.input_name)
        G = nx.from_numpy_matrix(np.load(input))

    elif args.input_type == "txt":
        assert (not args.weighted)
        input = '%s/../graph/%s_adj.txt' % (c, args.input_name)
        with open(input, "rb") as g:
            G = nx.read_adjlist(g)
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    print 'Number of walks', len(walks)
    print 'An example walk', walks[7]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size,
                     min_count=0, sg=1, workers=args.workers, iter=args.iter)
    emb_file = '%s/../emb/%s.emb' % (c, args.input_name)
    model.wv.save_word2vec_format(emb_file)
    print 'args.window_size', args.window_size
    convert_embed_to_np(emb_file, '%s/../emb/%s_emb_iter_%s_p_%s_q_%s_walk_%s_win_%s.npy' % (c, args.input_name, args.iter, args.p, args.q, args.num_walks, args.window_size))

    return


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    print 'Reading graph'
    nx_G = read_graph()
    print 'Creating node2vec graph'
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    print 'Preprocessing'
    G.preprocess_transition_probs()
    print 'Generating walks'
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    print 'Learning embeddings'
    learn_embeddings(walks)


if __name__ == "__main__":
    args = parse_args()
    main(args)
