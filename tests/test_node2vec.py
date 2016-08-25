#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
tests for node2vec
"""

import os
from node2vec import node2vec
from node2vec.node2vec import read_graph,learn_embeddings
import tempfile
import numpy

datafile = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'graph', 'karate.edgelist')
outfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'emb', 'karate.emb')

def test_node2vec_datafile():
    print('datafile:%s'%datafile)
    assert os.path.exists(datafile)
    print('outfile:%s'%outfile)
    assert os.path.exists(outfile)
    
def test_node2vec_run():
    # use defaults from main script
    weighted=False
    directed=False
    p=1
    q=1
    dimensions=128
    window_size=10
    workers=8
    iter=1
    num_walks=10
    walk_length=10
    test_outfile='/tmp/node2vec_test.txt'
    nx_G = read_graph(datafile,weighted,directed)
    G = node2vec.Graph(nx_G, directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    learn_embeddings(walks,test_outfile,dimensions,
                  window_size,workers,iter)
