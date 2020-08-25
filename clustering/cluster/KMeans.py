import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Class that implements the K-means clustering algorithm.
# K-means is an unsupervised algorithm. Meaning in takes data that doesn't really have any pertinent information.
# The goal of an unsupervised algorithm is to make sense out of the data
# as opposed to supervised learning, where there is a relationship between the data.
# Good articles for K-means:
# https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
# https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a

class KMeans:
    coef_ = np.array([]) # Estimated coefficients
    centroids = np.array([]) # Coordinates of centroids
    def __init__(self, K):
        self.K = K

    def fit(self, X, y):
        choose_random_centroid()

    def get_params(self):
        pass

    def predict(self, X):
        pass

    def score(self, predictions, y, sample_weights = None):
        pass

    def msee(self, actual, predicted):
        pass

    # Computes random point from X and uses it as initial centroid
    # ***CAUTION*** You don't know which is actually x and which is actually y. So you need to fix this later

    def choose_random_centroid(self,X):
        for i in range(self.K):
            x = np.random.choice(X[0],1)[0]
            y = np.random.choice(X[1],1)[0]
            self.centroids = np.append(self.centroids,[x,y],axis=1)

kmeans = KMeans(2)
kmeans.choose_random_centroid([[1,2,3],[4,5,6]])
print(kmeans.centroids)