import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, silhouette_samples, normalized_mutual_info_score, \
    fowlkes_mallows_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
        kmeans = KMeans(n_clusters=n_clusters, random_state=100)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        #print(f'Number of Clusters: {n_clusters}, Silhouette Score: {silhouette_avg}')

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = n_clusters

    print(f'Best Number of Clusters: {best_num_clusters}, Best Silhouette Score: {best_silhouette_score}')
    return best_num_clusters


def pca_dimension_reduction(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    return X_reduced


def kmeans(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels


def evaluate_clustering(X, true_labels, predicted_labels):
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

def dbscan_clustering(X, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print(f'Estimated number of clusters: {n_clusters}')
    print(f'Estimated number of noise points: {n_noise}')

    return cluster_labels

# Add DBSCAN to the model functions dictionary
def unsup_models(X, c=None, n=2, eps=0.5, min_samples=5):
    model_functions = {
        'kmeans': lambda: kmeans(X, n_clusters=n),
        'pca': lambda: pca_dimension_reduction(X, n_components=2),
        'dbscan': lambda: dbscan_clustering(X, eps=eps, min_samples=min_samples)
    }

    result = model_functions[c]()
    return result


def visualize_clusters(X, labels, class_names):
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='inferno', marker='o', edgecolor='k', s=50)


    handles, _ = scatter.legend_elements()
    unique_labels = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.get_cmap()(scatter.norm(int(label))),
                          markersize=10, markeredgecolor='k') for label in unique_labels]
    legend = plt.legend(handles, [class_names[int(label)] for label in unique_labels], title="Classes",
                        loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title('Clusters Visualization')
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make space for the legend
    plt.show()


def visualize_clusters_tsne(X, cluster_labels, actual_labels, class_names):
    # Apply t-SNE reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    X_reduced = tsne.fit_transform(X)

    plt.figure(figsize=(18, 9))  # Adjust the figure size as necessary
    # Use a distinct colormap and plot
    cmap = plt.get_cmap('viridis', len(np.unique(cluster_labels)))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap=cmap, edgecolor='k', s=60)

    unique_labels = np.unique(cluster_labels)
    cluster_to_class = {}
    for label in unique_labels:
        indices = np.where(cluster_labels == label)
        most_common_label = np.bincount(actual_labels[indices]).argmax()
        cluster_to_class[label] = class_names[most_common_label]

    # Customize the ticks to make grid less dense
    plt.xticks(ticks=np.arange(min(X_reduced[:, 0]), max(X_reduced[:, 0])+1, 10))
    plt.yticks(ticks=np.arange(min(X_reduced[:, 1]), max(X_reduced[:, 1])+1, 10))

    # Reduce grid line visibility or remove grid
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Lighter grid

    # Legend outside the plot
    handles = [plt.Line2D([0], [0], marker='o', color=cmap(i / max(cluster_labels)), linestyle='', markersize=10, markeredgecolor='k') for i in unique_labels]
    labels = [f"{cluster_to_class[label]} ({np.count_nonzero(cluster_labels == label)})" for label in unique_labels]
    plt.legend(handles, labels, title='Classes', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')


    plt.title('Enhanced t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.show()

def visualize_clusters_tsne_label(X, cluster_labels, actual_labels, class_names):
    # Apply t-SNE reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    X_reduced = tsne.fit_transform(X)

    plt.figure(figsize=(18, 9))  # Adjust the figure size as necessary
    # Use a distinct colormap and plot
    cmap = plt.get_cmap('viridis', len(np.unique(cluster_labels)))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap=cmap, edgecolor='k', s=60)

    unique_labels = np.unique(cluster_labels)
    cluster_to_class = {}
    for label in unique_labels:
        indices = np.where(cluster_labels == label)
        most_common_label = np.bincount(actual_labels[indices]).argmax()
        cluster_to_class[label] = class_names[most_common_label]

    # Customize the ticks to make grid less dense
    plt.xticks(ticks=np.arange(min(X_reduced[:, 0]), max(X_reduced[:, 0])+1, 10))
    plt.yticks(ticks=np.arange(min(X_reduced[:, 1]), max(X_reduced[:, 1])+1, 10))

    # Reduce grid line visibility or remove grid
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Lighter grid

    # Legend outside the plot
    handles = [plt.Line2D([0], [0], marker='o', color=cmap(i / max(cluster_labels)), linestyle='', markersize=10, markeredgecolor='k') for i in unique_labels]
    labels = [f"{cluster_to_class[label]} ({np.count_nonzero(cluster_labels == label)})" for label in unique_labels]
    plt.legend(handles, labels, title='Classes', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')
    centroids = {}
    for label in unique_labels:
        indices = np.where(cluster_labels == label)
        if len(indices[0]) > 0:
            most_common_label = np.bincount(actual_labels[indices]).argmax()
            cluster_to_class[label] = class_names[most_common_label]
            cluster_center = np.mean(X_reduced[indices], axis=0)
            centroids[label] = cluster_center
        else:
            cluster_to_class[label] = "Unknown"

    # Place text annotations
    for label, centroid in centroids.items():
        plt.text(centroid[0], centroid[1], cluster_to_class[label], fontsize=9, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    plt.title('Enhanced t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.show()

def visualize_clusters_with_pca(X, cluster_labels, true_labels, class_names):
    # Reduce dimensionality with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Map each cluster label to the most frequent actual label in that cluster
    cluster_to_label_mapping = {}
    for cluster in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cluster)
        actual_labels_in_cluster = true_labels[indices]
        most_common_label = np.bincount(actual_labels_in_cluster).argmax()
        cluster_to_label_mapping[cluster] = class_names[most_common_label]

    # Create the scatter plot
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', marker='o', edgecolor='k', s=50)

    # Creating custom legend with manual mapping
    unique_clusters = np.unique(cluster_labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / max(unique_clusters)), markersize=10, markeredgecolor='k') for i in unique_clusters]
    new_labels = [cluster_to_label_mapping[cluster] for cluster in unique_clusters]
    plt.legend(handles, new_labels, title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title('Corrected PCA Clusters Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True)
    plt.show()



