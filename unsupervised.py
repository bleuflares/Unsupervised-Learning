import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn import metrics
from sklearn import random_projection, neighbors, neural_network

def discrete_convert(l):
    n = []
    for x in l:
        if x < 0.25:
            n.append(0)
        elif x < 0.5:
            n.append(1)
        elif x < 0.75:
            n.append(2)
        else:
            n.append(3)
    return n

def euc_dist(arr1, arr2):
    diff = np.array(arr1) - np.array(arr2)
    return np.sum(np.dot(diff, diff))

def get_admission_input():
    file = open("dataset/Admission_Predict_Ver1.1.csv", 'r')
    return np.loadtxt(file, delimiter=",", skiprows=1)

def get_accident_input():
    file = open("dataset/Accident.csv", 'r')
    return np.loadtxt(file, delimiter=",", skiprows=1)

def kmeans(t_x, t_y, v_x, v_y):
    print("kmeans")
    kmeans = KMeans(n_clusters=4, random_state=0).fit(t_x)
    train_score = metrics.accuracy_score(t_y, kmeans.predict(t_x))
    v_score = metrics.accuracy_score(v_y, kmeans.predict(v_x))
    print(train_score, v_score)

def em(t_x, t_y, v_x, v_y):
    print("em")
    cov_type = ['full', 'tied', 'diag', 'spherical']
    for cov in cov_type:
        em = GaussianMixture(n_components=2, covariance_type=cov).fit(t_x)
        print(cov, metrics.accuracy_score(t_y, em.predict(t_x)))

def pure(t_x, t_y, v_x, v_y):
    print("control group")
    result = []
    model = neighbors.KNeighborsClassifier(10, weights='distance')    
    model.fit(t_x, t_y)
    print(metrics.accuracy_score(v_y, model.predict(v_x)))
    

def pca(t_x, t_y, v_x, v_y):
    print("PCA")
    result = []
    component_num = [2, 4, 6, 8]
    model = neighbors.KNeighborsClassifier(10, weights='distance')
    
    for i in component_num:
        pca = PCA(n_components=i)
        pca.fit(t_x)
        t_x_reduced = pca.transform(t_x)
        v_x_reduced = pca.transform(v_x)
        model.fit(t_x_reduced, t_y)
        result.append(metrics.accuracy_score(v_y, model.predict(v_x_reduced)))
    print(result)

def ica(t_x, t_y, v_x, v_y):
    print("ICA")
    result = []
    component_num = [2, 4, 6, 8]
    model = neighbors.KNeighborsClassifier(10, weights='distance')
    
    for i in component_num:
        ica = FastICA(n_components=i)
        ica.fit(t_x)
        t_x_reduced = ica.transform(t_x)
        v_x_reduced = ica.transform(v_x)
        model.fit(t_x_reduced, t_y)
        result.append(metrics.accuracy_score(v_y, model.predict(v_x_reduced)))
    print(result)

def rp(t_x, t_y, v_x, v_y):
    print("Randomized Projection")
    result = []
    component_num = [2, 4, 6, 8]
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
    print(result)

def dim_reduct(name, t_x, t_y, v_x, v_y):
    print("dimentionality reduction")
    if name == 'kmeans':
        model = KMeans(n_clusters=4, random_state=0)
    elif name == 'em':
        model = GaussianMixture(n_components=2, covariance_type='full')
    
    comp = [2, 4, 6, 8]
    methods = ['PCA', 'ICA', 'RP']

    file = open(name + "dim_reduct.csv", "w")
    result = ""
    result_v  = ""
    
    for j in comp:
        print(j)
        for name in methods:
            temp = []
            temp_v = []
            if name == 'RP':
                iters = 20
            else:
                iters = 1
            
            for it in range(iters):
                if name == 'PCA':
                    method = PCA(n_components=j)
                elif name == 'ICA':
                    method = FastICA(n_components=j)
                elif name == 'RP':
                    method = random_projection.GaussianRandomProjection(n_components=j)

                t_x_reduced = method.fit_transform(t_x)
                v_x_reduced = method.fit_transform(v_x)
                model.fit(t_x_reduced)
                    
                acc = metrics.accuracy_score(t_y, model.predict(t_x_reduced))
                acc_v = metrics.accuracy_score(v_y, model.predict(v_x_reduced))
                
                temp.append(acc)
                temp_v.append(acc_v)
                
            result += str(np.mean(temp)) + ", "
            result_v += str(np.mean(temp_v)) + ", "
        result +=  "\n"
        result_v +=  "\n"
    file.write(result)
    file.write(result_v)
    file.close()

def dim_reduct_nn(t_x, t_y, v_x, v_y):
    print("dimentionality reduction nn")
    model = neural_network.MLPClassifier(hidden_layer_sizes=(5,5))
    
    comp = [2, 4, 6, 8]
    methods = ['PCA', 'ICA', 'RP']

    file = open("dim_reduct_nn.csv", "w")
    result = ""
    result_v  = ""
    
    for j in comp:
        print(j)
        for name in methods:
            temp = []
            temp_v = []
            if name == 'RP':
                iters = 20
            else:
                iters = 1
            
            for it in range(iters):
                if name == 'PCA':
                    method = PCA(n_components=j)
                elif name == 'ICA':
                    method = FastICA(n_components=j)
                elif name == 'RP':
                    method = random_projection.GaussianRandomProjection(n_components=j)

                t_x_reduced = method.fit_transform(t_x)
                v_x_reduced = method.fit_transform(v_x)
                model.fit(t_x_reduced, t_y)
                    
                acc = metrics.accuracy_score(t_y, model.predict(t_x_reduced))
                acc_v = metrics.accuracy_score(v_y, model.predict(v_x_reduced))
                
                temp.append(acc)
                temp_v.append(acc_v)
                
            result += str(np.mean(temp)) + ", "
            result_v += str(np.mean(temp_v)) + ", "
        result +=  "\n"
        result_v +=  "\n"
    file.write(result)
    file.write(result_v)
    file.close()

def cluster_nn(name, t_x, t_y, v_x, v_y):
    if name == 'kmeans':
        cluster = KMeans(n_clusters=4, random_state=0)
    elif name == 'em':
        cluster = GaussianMixture(n_components=2, covariance_type='full')
    print("cluster nn")
    model = neural_network.MLPClassifier(hidden_layer_sizes=(5,5))
    
    comp = [2, 4, 6, 8]
    methods = ['PCA', 'ICA', 'RP']

    file = open(name + "cluster_nn.csv", "w")
    result = ""
    result_v  = ""
    
    for j in comp:
        print(j)
        for name in methods:
            temp = []
            temp_v = []
            if name == 'RP':
                iters = 20
            else:
                iters = 1
            
            for it in range(iters):
                if name == 'PCA':
                    method = PCA(n_components=j)
                elif name == 'ICA':
                    method = FastICA(n_components=j)
                elif name == 'RP':
                    method = random_projection.GaussianRandomProjection(n_components=j)

                t_x_reduced = method.fit_transform(t_x)
                v_x_reduced = method.fit_transform(v_x)
                cluster.fit(t_x_reduced)
                clustered = cluster.predict(t_x_reduced)
                clustered_v = cluster.predict(v_x_reduced)
                clustered = clustered.reshape(clustered.shape[0], 1)
                clustered_v = clustered_v.reshape(clustered_v.shape[0], 1)

                t_x_new = np.hstack([t_x_reduced, clustered])
                v_x_new = np.hstack([v_x_reduced, clustered_v])

                model.fit(t_x_new, t_y)
                    
                acc = metrics.accuracy_score(t_y, model.predict(t_x_new))
                acc_v = metrics.accuracy_score(v_y, model.predict(v_x_new))
                
                temp.append(acc)
                temp_v.append(acc_v)
                
            result += str(np.mean(temp)) + ", "
            result_v += str(np.mean(temp_v)) + ", "
        result +=  "\n"
        result_v +=  "\n"
    file.write(result)
    file.write(result_v)
    file.close()

if __name__ == "__main__":
    print("small input")
    array = get_admission_input()
    x = array[:, :8]
    x = (x / x.max(axis=0))
    tx = x[:len(array) * 9 / 10]
    ty = discrete_convert(array[:len(array) * 9 / 10, 8])
    vx = x[len(array) * 9 / 10:]
    vy = discrete_convert(array[len(array) * 9 / 10:, 8])
    kmeans(tx, ty, vx, vy)
    em(tx, ty, vx, vy)
    pure(tx, ty, vx, vy)
    pca(tx, ty, vx, vy)
    ica(tx, ty, vx, vy)
    rp(tx, ty, vx, vy)
    dim_reduct('kmeans', tx, ty, vx, vy)
    dim_reduct('em', tx, ty, vx, vy)
    dim_reduct_nn(tx, ty, vx, vy)
    cluster_nn('kmeans', tx, ty, vx, vy)
    cluster_nn('em', tx, ty, vx, vy)


    print("big input")
    array = get_accident_input()
    x = array[:, :13]
    x = (x / x.max(axis=0))
    tx = x[:len(array) * 9 / 10]
    ty = array[:len(array) * 9 / 10, 13]
    vx = x[len(array) * 9 / 10:]
    vy = array[len(array) * 9 / 10:, 13]
    kmeans(tx, ty, vx, vy)
    em(tx, ty, vx, vy)
    pure(tx, ty, vx, vy)
    pca(tx, ty, vx, vy)
    ica(tx, ty, vx, vy)
    rp(tx, ty, vx, vy)
    dim_reduct('kmeans', tx, ty, vx, vy)
    dim_reduct('em', tx, ty, vx, vy)
    dim_reduct_nn(tx, ty, vx, vy)
    cluster_nn('kmeans', tx, ty, vx, vy)
    cluster_nn('em', tx, ty, vx, vy)
