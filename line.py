import numpy as np

from node_class import read_node_label, Classifier
from GraphEmbedding.ge import LINE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('GraphEmbedding/data/wiki/wiki_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings)
    clf.train_and_evaluate_split(X, Y, tr_frac)



if __name__ == "__main__":
    G = nx.read_edgelist('GraphEmbedding/data/wiki/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = LINE(G, embedding_size=128, order='first')
    model.train(batch_size=1024, epochs=50, verbose=2)
    embeddings = model.get_embeddings()

    #print(embeddings)

    evaluate_embeddings(embeddings)
