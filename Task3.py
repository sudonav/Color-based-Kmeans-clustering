
# coding: utf-8

# In[ ]:


UBIT = 'nramanat'
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))
import cv2
import matplotlib.pyplot as plt


# In[ ]:


def getEuclideanDistance(A, B):
    return np.linalg.norm(A - B)


# In[ ]:


color_dict = {0: 'r', 1: 'g', 2: 'b'}


# In[ ]:


X = np.asarray([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]])


# In[ ]:


def stopping_cond(mu, old_mu):
    return np.array_equal(mu, old_mu)

def get_mu(X, k):
    mu = X.copy()
    np.random.shuffle(mu)
    return mu[:k]

def KMeans(X, mu, k=3):
    indx = np.array([0] * len(X))
    
    m, n = len(mu), len(X)
    for i in range(n):
        dist = [] 
        for j in range(m):
            dist.append(getEuclideanDistance(mu[j], X[i]))
        indx[i] = np.argmin(dist)
    
    for i in range(m):
        points = X[indx == i]
        if len(points) == 0:
            continue
        mu[i] = np.mean(points, axis=0)
        
    return mu, indx


# In[ ]:


mu = np.asarray([[6.2,3.2],[6.6,3.7],[6.5,3.0]])
k = 3

old_mu = np.ones((k, X.shape[1])) * (1e5)
indx = None

if mu is None:
    mu = get_mu(X, k)

while not stopping_cond(mu, old_mu):
    old_mu = np.copy(mu)
    mu, indx = KMeans(X, mu, k)
    figure = plt.figure()
    x,y = X.T
    for i in range(len(X)):
        plt.scatter(X[i, 0], X[i, 1], marker='^', facecolors=color_dict[indx[i]])
    for i in range(len(mu)):
        plt.scatter(mu[i, 0], mu[i, 1], facecolors=color_dict[i])
    plt.show()


# In[ ]:


X = cv2.imread("baboon.jpg")
X = np.array(X, dtype=np.float64) / 255
width, height, depth = orig_shape = tuple(X.shape)
X = X.reshape((-1, 3))


# In[ ]:


K = [3,5,10,20]

for k in K :
    mu = None
    old_mu = np.ones((k, 3)) * (1e5)

    if mu is None:
        mu = get_mu(X, k)

    while not stopping_cond(mu, old_mu):
        old_mu = np.copy(mu)
        mu, indx = KMeans(X, mu, k)

    depth = mu.shape[1]
    output = np.zeros((width, height, depth))
    label = 0
    for i in range(width):
        for j in range(height):
            output[i][j] = mu[indx[label]]
            label += 1
    output = output * 255
    fileName = "task3_baboon_"+str(k)+".jpg"
    cv2.imwrite(fileName,output)

