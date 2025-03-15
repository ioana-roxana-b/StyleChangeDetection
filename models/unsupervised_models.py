from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def elbow_method(X, max_clusters = 30):
    """
       Uses the Elbow Method to identify the optimal number of clusters for KMeans.
       Computes and plots the within-cluster sum of squares (WCSS) for a range of cluster counts.
       Params:
           X (array-like): Input data for clustering.
           max_clusters (int): Maximum number of clusters to consider.
       Returns:
           None: Displays the Elbow Method plot to help determine the optimal number of clusters.
    """

    max_clusters = min(max_clusters, X.shape[0])
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state = 42, n_init = 10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Determining Optimal Number of Clusters')
    plt.show()


def grid_search_kmeans(X, min_clusters = 2, max_clusters = 30):
    """
       Uses grid search to find the optimal number of clusters for KMeans based on the silhouette score.
       Params:
           X (array-like): Input data for clustering.
           min_clusters (int): Minimum number of clusters to consider.
           max_clusters (int): Maximum number of clusters to consider.
       Returns:
           int: Best number of clusters based on the silhouette score.
    """
    max_clusters = min(max_clusters, X.shape[0] - 1)  # Ensure max_clusters does not exceed number of samples - 1
    best_num_clusters = min_clusters
    best_silhouette_score = -1

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state = 42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        # print(f'Number of Clusters: {n_clusters}, Silhouette Score: {silhouette_avg}')

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = n_clusters

    print(f'Best Number of Clusters: {best_num_clusters}, Best Silhouette Score: {best_silhouette_score}')
    return best_num_clusters


def pca_dimension_reduction(X, n_components = 2):
    pca = PCA(n_components = n_components)
    X_reduced = pca.fit_transform(X)

    return X_reduced


def kmeans(X, n_clusters = 2):
    kmeans = KMeans(n_clusters = n_clusters, random_state = 42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels


def evaluate_clustering(X, true_labels, predicted_labels):
    """
       Evaluates clustering performance using multiple metrics: silhouette score, ARI, DB index, CH index, NMI, and FMI.
       Params:
           X (array-like): Input data used for clustering.
           true_labels (array-like): True class labels for the data.
           predicted_labels (array-like): Cluster labels predicted by the model.
       Returns:
           None: Prints the evaluation metrics to the console.
    """
    silhouette_avg = silhouette_score(X, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)

    db_index = davies_bouldin_score(X, predicted_labels)
    ch_index = calinski_harabasz_score(X, predicted_labels)

    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    fmi = fowlkes_mallows_score(true_labels, predicted_labels)

    print(f'Silhouette Score: {silhouette_avg}')
    print(f'Adjusted Rand Index: {ari}')
    print(f'Davies-Bouldin Index: {db_index}')
    print(f'Calinski-Harabasz Index: {ch_index}')
    print(f'Normalized Mutual Information: {nmi}')
    print(f'Fowlkes-Mallows Index: {fmi}')


def dbscan_clustering(X, eps = 0.5, min_samples = 5):
    """
       Applies DBSCAN clustering to the input data and returns the cluster labels.
       Params:
           X (array-like): Input data for clustering.
           eps (float): Maximum distance between two points to be considered neighbors.
           min_samples (int): Minimum number of points to form a dense region.
       Returns:
           array-like: Cluster labels for each data point. Noise points are labeled as -1.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print(f'Estimated number of clusters: {n_clusters}')
    print(f'Estimated number of noise points: {n_noise}')

    return cluster_labels

def unsup_models(X, c = None, n = 2, eps = 0.5, min_samples = 5):
    """
        Applies a specified unsupervised clustering or dimensionality reduction technique.
        Params:
            X (array-like): Input data for clustering or dimensionality reduction.
            c (str): Name of the method to apply ('kmeans', 'pca', or 'dbscan').
            n (int): Number of clusters for KMeans or number of components for PCA.
            eps (float): Maximum distance parameter for DBSCAN.
            min_samples (int): Minimum sample size parameter for DBSCAN.
        Returns:
            array-like: Output from the specified method (cluster labels or reduced dimensions).
    """
    model_functions = {
        'kmeans': lambda: kmeans(X, n_clusters = n),
        'pca': lambda: pca_dimension_reduction(X, n_components = 2),
        'dbscan': lambda: dbscan_clustering(X, eps = eps, min_samples = min_samples)
    }

    result = model_functions[c]()
    return result


