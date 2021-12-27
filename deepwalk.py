import numpy as np
import networkx as nx
import csrgraph as cg
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
    i = 0
    walks = []

    for i in range(gamma):
        np.random.shuffle(V) 
        for v in V:
            i += 1
            if(i%1024==0):
                print(i)
            W_v = RandomWalk(G,v,t)
            walks.append(W_v)

    model = Word2Vec(walks, size=d, window=w, min_count=0, sg=1, hs=1, workers=4, alpha=0.25, min_alpha=0.1)

    return model

d = 128 # embedding dimension

G = nx.read_edgelist("datasets/youtube.txt")
G = G.to_undirected()
"""
model = DeepWalk(G, w=10, d=d, gamma=80, t=40)
print(model)
"""

G_cg = cg.csrgraph(G, threads=12) 
walks = G_cg.random_walks(walklen=10, # length of the walks
                epochs=80, # how many times to start a walk from each node
                start_nodes=None, # the starting node. It is either a list (e.g., [2,3]) or None. If None it does it on all nodes and returns epochs*G.number_of_nodes() walks
                return_weight=1.,
                neighbor_weight=1.)

print(len(walks))
print(len(walks[0]))
print(walks[0])


model = Word2Vec(walks, vector_size=d, window=10, min_count=0, sg=1, hs=1, workers=4, alpha=0.25, min_alpha=0.1)
print(model)