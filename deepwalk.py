import numpy as np
import networkx as nx
from gensim.models import Word2Vec

def RandomWalk(G, v, t):
    path = np.zeros(t+1)
    path[0] = v

    for i in range(t):
        neighbors = np.array([n for n in G.neighbors(v)])
        v = np.random.choice(neighbors)
        path[i+1] = v

    return path

"""
Inputs:
    G = Graph G(V,E)
    w = window size
    d = embedding size
    gamma = walks per vertex
    t = walk length
Output:
    phi = matrix of vertex representations of size |V| x d 
"""
def DeepWalk(G, w, d, gamma, t):

    vertex_count = G.number_of_nodes()

    # initilization
    phi = np.random.standard_normal(size=(vertex_count,d))
    V = np.array(list(G.nodes(data=False)))

    #  build a binary tree T from V - skip for now

    walks = []

    for i in range(gamma):
        np.random.shuffle(V)
        for v in V:
            W_v = RandomWalk(G,v,t)
            walks.append(W_v)

    model = Word2Vec(walks, size=d, window=w, min_count=0, sg=1, hs=1, workers=4, alpha=0.25, min_alpha=0.1)

    return model

d = 128 # embedding dimension

G = nx.read_edgelist("datasets/youtube.txt")
G = G.to_undirected()

model = DeepWalk(G, w=10, d=d, gamma=80, t=40)
print(model)