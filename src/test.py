import numpy as np
import networkx as nx
import node2vec_ms
import node2vec_revers
import node2vec
from gensim.models import Word2Vec

from src.distance import get_matrixs, calc_matrix_norm

config = {
    'input': '../graph/lesmis.edgelist',
    # 'output': '../emb/lesmis{}.emb',
    'dimensions': 16,
    'walk_length': 10,
    'num_walks': 5,
    'window_size': 10,
    'iter': 10,
    'workers': 8,
    'p': 1,
    'q': 2,
    'weighted': False,
    'directed': False,
    'unweighted': True

}


def read_graph(config):
    '''
    Reads the input network in networkx.
    '''
    if config['weighted']:
        G = nx.read_edgelist(config['input'], nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(config['input'], nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not config['directed']:
        G = G.to_undirected()

    return G


def learn_embeddings(walks, config):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=config['dimensions'], window=config['window_size'], min_count=0, sg=1,
                     workers=config['workers'], iter=config['iter'])
    model.save_word2vec_format(config['output'])

    return


def test(config, impl):
    nx_G = read_graph(config)
    G = impl.Graph(nx_G, config['directed'], config['p'], config['q'])
    G.preprocess_transition_probs()
    walks = G.simulate_walks(config['num_walks'], config['walk_length'])
    print "LEN: " + str(len(walks))
    learn_embeddings(walks, config)


test_count = 3

output_schema = '../emb/lesmis{}.emb',

outputs_ms = [('../emb/lesmis' + str(i + 1) + '.emb', "MS_" + str(i+1)) for i in xrange(test_count)]
outputs_base = [('../emb/lesmis' + str(i + 1) + '.emb', "base_" + str(i+1)) for i in xrange(test_count, 2*test_count)]

for i in xrange(test_count):
    config['output'] = '../emb/lesmis' + str(i + 1) + '.emb'
    test(config, node2vec_ms)

for i in xrange(test_count, 2 * test_count):
    config['output'] = '../emb/lesmis' + str(i + 1) + '.emb'
    test(config, node2vec_revers)

for i in xrange(2*test_count, 3 * test_count):
    config['output'] = '../emb/lesmis' + str(i + 1) + '.emb'
    test(config, node2vec)


calc_matrix_norm(get_matrixs('../emb/lesmis{}.emb', 3*test_count))