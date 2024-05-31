import sklearn.cluster as skcls
from sklearn.neighbors import NearestCentroid
import sklearn.metrics as skmtr
import pandas as pd

def hierarchical(data, ncluster, metric, link):
    hierarki = skcls.AgglomerativeClustering(n_clusters=ncluster, metric=metric, linkage=link, compute_distances=True)
    hierarki_c = hierarki.fit(data)  # Fitting
    return hierarki_c

def kmeans(data, ncluster, initialization):
    km = skcls.KMeans(n_clusters=ncluster, init=initialization, n_init="auto")
    km_c = km.fit(data)  # Fitting
    return km_c

def get_centroid(data, ncluster, metric, link):
    hierarki = skcls.AgglomerativeClustering(n_clusters=ncluster, metric=metric, linkage=link, compute_distances=True)
    y_predict = hierarki.fit_predict(data)  # Mendapatkan label cluster data centroid dari Hierarchical Agglomerative variasi 1
    clf = NearestCentroid(metric="euclidean")
    clff = clf.fit(data, y_predict)
    centroid_agglo = clff.centroids_
    return centroid_agglo

def get_dbi(data, label_cluster):
    dbi = skmtr.davies_bouldin_score(data, label_cluster)
    return dbi

def evaluation(data, hierarki, hierarki2, maks_cluster):
    hc_distance = {}
    hc2_distance = {}
    hc_dbi = {}
    hc2_dbi = {}

    kc_inertia = {}
    kc2_inertia = {}
    kc_dbi = {}
    kc2_dbi = {}

    for i in range(len(hierarki.distances_[::-1])):
        hc_distance.update({(i + 1): hierarki.distances_[::-1][i]})  # Agglomerative distances variasi 1
        hc2_distance.update({(i + 1): hierarki2.distances_[::-1][i]})  # Agglomerative distances variasi 2

    for i in range(2, maks_cluster+1):
        # Hierarchical variasi 1
        hr = hierarchical(data, i, "euclidean", "ward")
        hc_dbi.update({i: get_dbi(data, hr.labels_)})  # DBI Hierarchical variasi 1

        # Hierarchical variasi 2
        hr2 = hierarchical(data, i, "manhattan", "average")
        hc2_dbi.update({i: get_dbi(data, hr2.labels_)})  # DBI Hierarchical variasi 2

        # KMeans
        ## Inertia
        kmean = kmeans(data, i, "k-means++")
        kc_inertia.update({i: kmean.inertia_})  # KMeans inertia variasi 1

        centroid = get_centroid(data, i, "euclidean", "ward")
        kmean2 = kmeans(data, i, centroid)
        kc2_inertia.update({i: kmean2.inertia_})  # KMeans inertia variasi 2

        ## DBI
        dbi_kc = get_dbi(data, kmean.labels_)
        kc_dbi.update({i: dbi_kc})  # DBI KMeans variasi 1

        dbi_kc2 = get_dbi(data, kmean2.labels_)
        kc2_dbi.update({i: dbi_kc2})  # DBI KMeans variasi 2

    dbi = pd.DataFrame([hc_dbi, hc2_dbi, kc_dbi, kc2_dbi]).T
    inertia = pd.DataFrame([kc_inertia, kc2_inertia]).T
    distances = pd.DataFrame([hc_distance, hc2_distance]).T

    df_dbi = dbi.rename(columns={0: "Hierarchical DBI 1", 1: "Hierarchical DBI 2", 2: "KMeans DBI 1", 3: "KMeans DBI 2"})
    df_inertia = inertia.rename(columns={0:"KMeans Inertia 1", 1: "KMeans Inertia 2"})
    df_distances = distances.rename(columns={0: "Hierarchical Distance 1", 1: "Hierarchical Distance 2"}) 
    return df_dbi, df_inertia, df_distances
