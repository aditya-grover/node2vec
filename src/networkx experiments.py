# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:53:49 2021

@author: dmitr
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 09:55:19 2021

@author: dmitr
"""
import os
import numpy as np
import networkx as nx

os.chdir(f"c:/Users/dmitr/Documents/git/node2vec")

# x = nx.generators.social.les_miserables_graph
G = nx.generators.social.karate_club_graph
G().number_of_nodes()
G().number_of_edges()
list(G().edges)
G()[0]
G()[15]
G()[33]

import matplotlib.pyplot as plt
subax1 = plt.subplot()

nx.draw(G(), with_labels=True, font_weight='bold')
nx.draw_circular(G(), with_labels=True, font_weight='bold')
nx.draw_spectral(G(), with_labels=True, font_weight='bold')
nx.draw_shell(G(), with_labels=True, font_weight='bold')
# plt.show()
plt.savefig("images/path.png")


nx.readwrite.edgelist.write_edgelist(G = G(), path = "graph/g.edgelist", data = False)

# 