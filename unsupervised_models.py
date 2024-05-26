from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def pca_dimension_reduction(X, n_components = 2):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    return X_reduced

def kmeans(X, n_clusters = 2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans

def unsup_models(X, c=None, n=2):
    model_functions = {
        'kmeans': lambda: kmeans(X, n_clusters=n),
        'pca': lambda: pca_dimension_reduction(X, n_components=2),
    }

    result = model_functions[c]()

    return result

def visualize_clusters(c, X, labels):
    plt.figure(figsize=(8, 6))
    if c == 'kmeans':
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels.labels_, cmap='viridis', marker='o', edgecolor='k', s=50)
        plt.scatter(labels.cluster_centers_[:, 0], labels.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
    else:
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)

    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Clusters Visualization')
    plt.show()

