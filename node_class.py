
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import make_multilabel_classification

import numpy as np

from  GraphEmbedding.ge.classify import Classifier as C1



class Classifier(object):
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.mlb = MultiLabelBinarizer()
        self.model = MultiOutputClassifier(LogisticRegressionCV(cv=5, multi_class="ovr"))

    def train(self, X, y):
        X_embedding = [self.embeddings[x] for x in X]
        y_bin = self.mlb.transform(y)
        self.model.fit(X_embedding, y_bin)
    
    def predict(self, X):
        X_embedding = [self.embeddings[x] for x in X]
        self.model.predict(X_embedding)
        print(self.model.predict_proba(X_embedding)[:10])
        return self.model.predict(X_embedding)
    
    def train_and_evaluate_split(self, X, y, split):
        self.mlb.fit(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
        y_test_bin = self.mlb.transform(y_test)
        print(y_test_bin[:10])
        self.train(X_train, y_train)
        y_pred = self.predict(X_test)
        print()
        print(y_pred[:10])
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = f1_score(y_test_bin, y_pred, average=average)
        
        print(results)
        return results

def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y

if __name__ == '__main__':
    X,y = make_multilabel_classification()
    print(y)
    m=MultiOutputClassifier(LogisticRegressionCV(cv=5, multi_class="ovr")).fit(X,y)
    print(m.predict_proba(X)[0])
    print(m.predict(X)[0])
    print()
    """
    m1=LogisticRegression(multi_class="ovr").fit(X,y)
    print(m1.predict_proba(X)[0])
    print(m1.predict(X)[0])
    y = np.array(y)
    numLabels = np.sum(y, axis=1)
    np.argpartition(a, -4)[-4:]"""
    m1 = C1(embeddings=dict(enumerate(X)), clf=LogisticRegression())
    y = np.array(y)
    y_norm = []
    for i in y:
        temp = []
        for j in range(len(i)):
            if i[j] > 0:
                temp.append(j)
        
        y_norm.append(temp)
    
    m1.split_train_evaluate([range(len(X))], y_norm, 0.8)



    print("Here we will have the classifier class")