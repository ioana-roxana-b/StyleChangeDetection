import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def elbow_method(X, max_clusters=30):
    max_clusters = min(max_clusters, X.shape[0])
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Determining Optimal Number of Clusters')
    plt.show()

def grid_search_kmeans(X, min_clusters=2, max_clusters=30):
    max_clusters = min(max_clusters, X.shape[0] - 1)  # Ensure max_clusters does not exceed number of samples - 1
    best_num_clusters = min_clusters
    best_silhouette_score = -1

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f'Number of Clusters: {n_clusters}, Silhouette Score: {silhouette_avg}')

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = n_clusters

    print(f'Best Number of Clusters: {best_num_clusters}, Best Silhouette Score: {best_silhouette_score}')
    return best_num_clusters

def pca_dimension_reduction(X, n_components = 2):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    return X_reduced

def kmeans(X, n_clusters = 2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans


def evaluate_clustering(X, true_labels, kmeans):
    predicted_labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)

    print(f'Silhouette Score: {silhouette_avg}')
    print(f'Adjusted Rand Index: {ari}')

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


    plt.title('Clusters Visualization')
    plt.show()


def visualize_clusters_v2(c, X, labels, class_names):
    plt.figure(figsize=(12, 6))  # Increased figure width for better legend visibility
    if c == 'kmeans':
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels.labels_, cmap='viridis', marker='o', edgecolor='k', s=50)
    else:
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)

    # Creating a legend with class names
    handles, _ = scatter.legend_elements()
    unique_labels = np.unique(labels.labels_ if c == 'kmeans' else labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.get_cmap()(scatter.norm(int(label))), markersize=10, markeredgecolor='k') for label in unique_labels]
    legend = plt.legend(handles, [class_names[int(label)] for label in unique_labels], title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title('Clusters Visualization')
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make space for the legend
    plt.show()


def visualize_clusters_v2_tsne(c, X, labels, class_names):
    # Dimensionality reduction with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(X)

    plt.figure(figsize=(12, 6))  # Increased figure width for better legend visibility

    # Determine the color mapping based on clustering method
    if c == 'kmeans':
        cluster_labels = labels.labels_
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap='viridis', marker='o',
                              edgecolor='k', s=50)
        plt.scatter(labels.cluster_centers_[:, 0], labels.cluster_centers_[:, 1], c='red', marker='X', s=300,
                    label='Centroids', edgecolor='k')
    else:
        cluster_labels = labels
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap='viridis', marker='o',
                              edgecolor='k', s=50)

    # Debug prints
    unique_labels = np.unique(cluster_labels)
    print(f"Unique Labels: {unique_labels}")
    print(f"Class Names: {class_names}")

    # Verify class names length
    if len(class_names) < len(unique_labels):
        raise ValueError(
            "The number of class names provided does not match the number of unique labels in the clusters.")

    # Creating a legend with class names
    handles, _ = scatter.legend_elements()
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.get_cmap()(scatter.norm(label)),
                          markersize=10, markeredgecolor='k') for label in unique_labels]
    legend = plt.legend(handles, [class_names[int(label)] for label in unique_labels], title="Classes",
                        loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title('Clusters Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for the legend
    plt.grid(True)  # Optional: add grid for better readability
    plt.show()
