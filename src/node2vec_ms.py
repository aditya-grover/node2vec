import numpy as np
import networkx as nx
import random
from collections import deque


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def draw_node(self, node, count, node_neighbors):
        result = {}
        for i in xrange(count):
            next = node_neighbors[alias_draw(self.alias_nodes[node][0], self.alias_nodes[node][1])]
            result[next] = result.get(next, 0) + 1
        return result

    def draw_edge(self, prev, node, count, node_neighbors):
        result = {}
        for i in xrange(count):
            next = node_neighbors[alias_draw(self.alias_edges[(prev, node)][0], self.alias_edges[(prev, node)][1])]
            result[next] = result.get(next, 0) + 1
        return result

    def node2vec_walk(self, walk_length, start_node, num_walks):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        walks = []
        queue = deque()
        queue.append([[start_node] for _ in xrange(num_walks)])
        while queue:
            cur_list = queue.pop()
            cur_walk = cur_list[0]
            cur = cur_walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(cur_walk) == 1:
                    drawn = self.draw_node(cur, len(cur_list), cur_nbrs)
                    for k, v in drawn.iteritems():
                        updated_walks = [cur_walk + [k] for _ in xrange(v)]
                        if len(updated_walks[0]) == walk_length:
                            walks += updated_walks
                        else:
                            queue.append(updated_walks)

                else:
                    drawn = self.draw_edge(cur_walk[-2], cur, len(cur_list), cur_nbrs)
                    for k, v in drawn.iteritems():
                        updated_walks = [cur_walk + [k] for _ in xrange(v)]
                        if len(updated_walks[0]) == walk_length:
                            walks += updated_walks
                        else:
                            queue.append(updated_walks)
            else:
                print("HANDLE NODE WITH NO NEIGHBORS")

        return walks

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # print 'Walk iteration:'
        # for walk_iter in range(num_walks):
        #     print str(walk_iter + 1), '/', str(num_walks)
        random.shuffle(nodes)
        for node in nodes:
            walks += self.node2vec_walk(walk_length=walk_length, start_node=node, num_walks=num_walks)

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

# print alias_setup([0.0164012 , 0.03986905, 0.07207824, 0.07763892, 0.06469869,
#        0.09192082, 0.05299032, 0.04480718, 0.00845884, 0.08852418,
#        0.00242594, 0.02412631, 0.11474021, 0.01500069, 0.0513678 ,
#        0.03225817, 0.04515068, 0.04732195, 0.02723862, 0.08298219])
# print alias_setup([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.55])
# (array([ 3,  3,  0,  2,  3,  4,  5,  3,  5,  6,  9, 12,  9, 12, 12, 19, 19,
#        19, 19, 14]), array([ 0.328024 ,  0.797381 ,  1.       ,  0.5584352,  0.9841082,
#         0.6901344,  0.6825412,  0.8961436,  0.1691768,  0.6227348,
#         0.0485188,  0.4825262,  0.8037324,  0.3000138,  0.7263882,
#         0.6451634,  0.9030136,  0.946439 ,  0.5447724,  0.6990322]))
