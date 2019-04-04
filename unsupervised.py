import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn import cross_validation, metrics
from sklearn import random_projection, neighbors, neural_network


def euc_dist(arr1, arr2):
    diff = np.array(arr1) - np.array(arr2)
    return np.sum(np.dot(diff, diff))

def get_admission_input():
    file = open("dataset/Admission_Predict_Ver1.1.csv", 'r')
    return np.loadtxt(file, delimiter=",", skiprows=1)

def kmeans(t_x, t_y, v_x, v_y):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(t_x)
    train_score = metrics.accuracy_score(t_y, kmeans.predict(t_x))
    test_score = metrics.accuracy_score(v_y, kmeans.predict(v_x))
    return train_score, test_score

def em(t_x, t_y, v_x, v_y):
	cov_type = ['full', 'tied', 'diag', 'spherical']
    for cov in cov_type:
        em = GaussianMixture(n_components=2, covariance_type=cov).fit(t_x)
        print(d, metrics.accuracy_score(t_y, em.predict(t_x)))

def pca(t_x, t_y, v_x, v_y):
    result = []
    component_num = [2, 5, 10, 25, 50]
    model = neighbors.KNeighborsClassifier(10, weights='distance')
    
    for i in component_num:
        print(i)
        pca = PCA(n_components=i)
        pca.fit(t_x)
        t_x_reduced = pca.transform(t_x)
        v_x_reduced = pca.transform(v_x)
        model.fit(t_x_reduced, l.y)
        result.append(metrics.accuracy_score(v_y, model.predict(v_x_reduced)))
    return result

def pca(t_x, t_y, v_x, v_y):
    result = []
    component_num = [2, 5, 10, 25, 50]
    model = neighbors.KNeighborsClassifier(10, weights='distance')
    
    for i in component_num:
        print(i)
        pca = FastICA(n_components=i)
        pca.fit(t_x)
        t_x_reduced = pca.transform(t_x)
        v_x_reduced = pca.transform(v_x)
        model.fit(t_x_reduced, l.y)
        result.append(metrics.accuracy_score(v_y, model.predict(v_x_reduced)))
    return result

 def rp(t_x, t_y, v_x, v_y):
    result = []
    component_num = [2, 5, 10, 25, 50]
    model = neighbors.KNeighborsClassifier(10, weights='distance')
    
    for i in component_num:
        x = []
        for j in range(100):
            rp = random_projection.GaussianRandomProjection(n_components=i)
            rp.fit(t_x)
            t_x_reduced = rp.transform(t_x)
            v_x_reduced = rp.transform(v_x)
            model.fit(t_x_reduced, t_y)
            x.append(metrics.accuracy_score(v_y, model.predict(v_x_reduced)))
        result.append((np.mean(x), np.std(x)))

    return result

if __name__ == "__main__":
    print("small input")
    array = get_admission_input()
    x = array[:, :8]
    x = (x / x.max(axis=0))
    tx = x[:len(array) * 9 / 10]
    ty = array[:len(array) * 9 / 10, 8]
    vx = x[len(array) * 9 / 10:]
    vy = array[len(array) * 9 / 10:, 8]

