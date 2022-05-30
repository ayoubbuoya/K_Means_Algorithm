from numpy import *


class K_Means:
    def __init__(self, x, k=3, initial_centroids=None, iters=100):
        self.x = x
        self.k = k
        self.init_centroids = initial_centroids
        self.iters = iters
        if self.init_centroids is None:
            self.init_centroids = self.init_cents(self.x, self.k)

    def init_cents(self, x, k):
        m, n = x.shape
        centroids = zeros((k, n))
        ind = random.randint(0, m, k)

        for i in range(k):
            centroids[i, :] = x[ind[i], :]
        return centroids

    def find_closest(self, x, centroids):
        m, n = x.shape
        k = centroids.shape[0]
        ind = zeros(m)

        for i in range(m):
            min_dist = 90000000

            for j in range(k):
                dist = sum((x[i, :] - centroids[j, :]) ** 2)

                if dist < min_dist:
                    min_dist = dist
                    ind[i] = j

        return ind

    def compute_new_centroids(self, x, ind, k):
        m, n = x.shape
        centroids = zeros((k, n))

        for i in range(k):
            idx = where(ind == i)
            centroids[i, :] = sum(x[idx, :], axis=1) / len(idx[0])

        return centroids

    def run_k_means(self):
        m, n = self.x.shape
        self.centroids = self.init_centroids
        k = self.centroids.shape[0]
        ind = zeros(m)

        for i in range(self.iters):
            ind = self.find_closest(self.x, self.centroids)
            self.centroids = self.compute_new_centroids(self.x, ind, self.k)

        return ind, self.centroids
