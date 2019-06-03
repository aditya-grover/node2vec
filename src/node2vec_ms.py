import numpy as np
import networkx as nx
import random
from collections import deque
import random


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def draw_node(self, node, next_steps_len, node_neighbors):
        result = {}
        for k, v in next_steps_len.iteritems():
            result[k] = {}
            for i in xrange(v):
                next = node_neighbors[alias_draw(self.alias_nodes[node][0], self.alias_nodes[node][1])]
                result[k][next] = result[k].get(next, 0) + 1
        return result

    def draw_edge(self, node, next_steps_len, node_neighbors):
        result = {}
        for k, v in next_steps_len.iteritems():
            result[k] = {}
            for i in xrange(v):
                next = node_neighbors[alias_draw(self.alias_edges[(k, node)][0], self.alias_edges[(k, node)][1])]
                result[k][next] = result[k].get(next, 0) + 1
        return result

    def update_step(self, drawn, drop_set, walk_length, walks, visit):
        for walk_list in drop_set:
            prev = self.get_prev(walk_list[0])
            updated_walks = []
            for walk_tuple in walk_list:
                next_step = random.choice(drawn[prev].keys())
                updated_walks.append((walk_tuple[0] + (next_step,), walk_tuple[1]))
                drawn[prev][next_step] = drawn[prev][next_step] - 1
                if not drawn[prev][next_step]:
                    del drawn[prev][next_step]
            if len(updated_walks[0][0]) == walk_length:
                walks += [list(updated_walks[0][0]) for _ in xrange(len(updated_walks))]
            else:
                visit.add(tuple(updated_walks))

    def get_prev(self, walk):
        return walk[0][-2] if len(walk[0]) > 1 else -1

    def node2vec_walk(self, walk_length, start_nodes, num_walks):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        walks = []
        visit = set()
        visitNext = set()
        for i, start_node in enumerate(start_nodes):
            visit.add(tuple(((start_node,), i) for _ in xrange(num_walks)))
        while visit:
            while visit:
                cur_list = visit.pop()  # Tuple containing all identical walks
                cur_walk = cur_list[0]  # Single walk to act as a representative of whole progress so far
                cur_id = cur_walk[1]  # Id of the random walk
                cur = cur_walk[0][-1]  # The last node of the currently considered walk
                drop_set = set()
                drop_set.add(cur_list)  # Walks that should be removed from visit as they are processed now
                # dict with number of steps and previous node (if it's not the first iteration) per walk id
                # next_steps_len = {cur_id: (len(cur_list), cur_walk[0][-2] if len(cur_walk[0]) > 1 else -1)}
                next_steps_len = {self.get_prev(cur_walk): len(cur_list)}
                for walk_list in visit:
                    cur_walk_overlap = walk_list[0]
                    if cur_walk_overlap[0][-1] == cur:
                        drop_set.add(walk_list)
                        next_steps_len[self.get_prev(cur_walk_overlap)] = next_steps_len.get(
                            self.get_prev(cur_walk_overlap), 0) + len(walk_list)
                        # next_steps_len[walk_list[0][1]] = (
                        #     len(walk_list), cur_walk_overlap[0][-2] if len(cur_walk[0]) > 1 else -1)
                cur_nbrs = sorted(G.neighbors(cur))
                if len(cur_nbrs) > 0:
                    if len(cur_walk[0]) == 1:
                        drawn = self.draw_node(cur, next_steps_len, cur_nbrs)
                        self.update_step(drawn, drop_set, walk_length, walks, visitNext)
                    else:
                        drawn = self.draw_edge(cur, next_steps_len, cur_nbrs)
                        self.update_step(drawn, drop_set, walk_length, walks, visitNext)
                else:
                    print("HANDLE NODE WITH NO NEIGHBORS")  # Probably just continue

                visit.difference_update(drop_set)
            visit, visitNext = visitNext, visit
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
            walks += self.node2vec_walk(walk_length=walk_length, start_nodes=[node], num_walks=num_walks)

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
