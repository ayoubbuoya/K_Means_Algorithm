from K_means import K_Means
from scipy.io import loadmat

data = loadmat("data2.mat")
x = data["X"]


model = K_Means(x)
ind, centroids = model.run_k_means()

print(ind)
print(centroids)
