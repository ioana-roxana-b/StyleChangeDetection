from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as sk_tsne, Isomap, MDS
import numpy as np
#from openTSNE import TSNE
import matplotlib.pyplot as plt


def visualize_clusters_tsne(X, cluster_labels, actual_labels, class_names):
    """
        Visualizes clusters using t-SNE and places annotations with class labels and centroids for each cluster.
        Params:
            X (array-like): Input data to be reduced using t-SNE.
            cluster_labels (array-like): Cluster labels for each data point.
            actual_labels (array-like): True class labels for the data.
            class_names (list): List of class names corresponding to each cluster label.
        Returns:
            None: Displays a t-SNE scatter plot of the clusters with annotated class labels.
    """
    # Apply t-SNE reduction
    tsne = sk_tsne(n_components=2, random_state=42, perplexity=50)
    X_reduced = tsne.fit_transform(X)

    plt.figure(figsize=(18, 9))  # Adjust the figure size as necessary
    # Use a distinct colormap and plot
    cmap = plt.get_cmap('viridis', len(np.unique(cluster_labels)))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap=cmap, edgecolor='k', s=60)

    unique_labels = np.unique(cluster_labels)
    centroids = {}
    cluster_to_class = {}

    for label in unique_labels:
        indices = np.where(cluster_labels == label)
        if len(indices[0]) > 0:
            most_common_label = np.bincount(actual_labels[indices]).argmax()
            cluster_to_class[label] = class_names[most_common_label]
            centroids[label] = np.mean(X_reduced[indices], axis=0)
        else:
            cluster_to_class[label] = "Unknown"

    # Customize the ticks to make grid less dense
    plt.xticks(ticks=np.linspace(np.min(X_reduced[:, 0]), np.max(X_reduced[:, 0]), num=10))
    plt.yticks(ticks=np.linspace(np.min(X_reduced[:, 1]), np.max(X_reduced[:, 1]), num=10))
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Lighter grid

    # Legend outside the plot
    handles = [plt.Line2D([0], [0], marker='o', color=cmap(i / max(cluster_labels)), linestyle='', markersize=10,
                          markeredgecolor='k') for i in unique_labels]
    labels = [f"{cluster_to_class[label]} ({np.count_nonzero(cluster_labels == label)})" for label in unique_labels]
    plt.legend(handles, labels, title='Classes', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Place text annotations
    for label, centroid in centroids.items():
        plt.text(centroid[0], centroid[1], cluster_to_class[label], fontsize=9, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    plt.title('Enhanced t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.show()


# def visualize_clusters_opentsne(X, cluster_labels, actual_labels, class_names):
#     """
#         Visualizes clusters using OpenTSNE and places annotations with class labels and centroids for each cluster.
#         Params:
#             X (array-like): Input data to be reduced using OpenTSNE.
#             cluster_labels (array-like): Cluster labels for each data point.
#             actual_labels (array-like): True class labels for the data.
#             class_names (list): List of class names corresponding to each cluster label.
#         Returns:
#             None: Displays an OpenTSNE scatter plot of the clusters with annotated class labels.
#     """
#     # Apply OpenTSNE reduction
#     tsne = TSNE(n_jobs=-1, perplexity=50, random_state=42)
#     X_reduced = tsne.fit(X)
#
#     plt.figure(figsize=(18, 9))  # Adjust the figure size as necessary
#     cmap = plt.get_cmap('viridis', len(np.unique(cluster_labels)))
#     scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap=cmap, edgecolor='k', s=60)
#
#     unique_labels = np.unique(cluster_labels)
#     centroids = {}
#     cluster_to_class = {}
#
#     for label in unique_labels:
#         indices = np.where(cluster_labels == label)
#         if len(indices[0]) > 0:
#             most_common_label = np.bincount(actual_labels[indices]).argmax()
#             cluster_to_class[label] = class_names[most_common_label]
#             centroids[label] = np.mean(X_reduced[indices], axis=0)
#         else:
#             cluster_to_class[label] = "Unknown"
#
#     # Customize the ticks to make grid less dense
#     plt.xticks(ticks=np.linspace(np.min(X_reduced[:, 0]), np.max(X_reduced[:, 0]), num=10))
#     plt.yticks(ticks=np.linspace(np.min(X_reduced[:, 1]), np.max(X_reduced[:, 1]), num=10))
#     plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Lighter grid
#
#     # Legend outside the plot
#     handles = [plt.Line2D([0], [0], marker='o', color=cmap(i / max(cluster_labels)), linestyle='', markersize=10,
#                           markeredgecolor='k') for i in unique_labels]
#     labels = [f"{cluster_to_class[label]} ({np.count_nonzero(cluster_labels == label)})" for label in unique_labels]
#     plt.legend(handles, labels, title='Classes', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')
#
#     # Place text annotations
#     for label, centroid in centroids.items():
#         plt.text(centroid[0], centroid[1], cluster_to_class[label], fontsize=9, ha='center', va='center',
#                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
#
#     plt.title('OpenTSNE Visualization of Clusters')
#     plt.xlabel('t-SNE Component 1')
#     plt.ylabel('t-SNE Component 2')
#     plt.tight_layout()
#     plt.show()


def visualize_clusters_pca(X, cluster_labels, true_labels, class_names):
    """
       Visualizes clusters using PCA for dimensionality reduction and maps clusters to the most frequent class label.
       Params:
           X (array-like): Input data to be reduced using PCA.
           cluster_labels (array-like): Cluster labels for each data point.
           true_labels (array-like): True class labels for the data.
           class_names (list): List of class names corresponding to each cluster label.
       Returns:
           None: Displays a PCA scatter plot of the clusters with mapped class labels.
    """
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
    cmap = plt.get_cmap('viridis', len(np.unique(cluster_labels)))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap=cmap, edgecolor='k', s=50)

    unique_clusters = np.unique(cluster_labels)
    handles = [plt.Line2D([0], [0], marker='o', color=cmap(i / max(cluster_labels)), linestyle='', markersize=10,
                          markeredgecolor='k') for i in unique_clusters]
    new_labels = [f"{cluster_to_label_mapping[cluster]} ({np.count_nonzero(cluster_labels == cluster)})"
                  for cluster in unique_clusters]
    plt.legend(handles, new_labels, title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title('PCA Visualization of Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.tight_layout()
    plt.show()


# MDS Visualization
def visualize_clusters_mds(X, cluster_labels, actual_labels, class_names):
    """
        Visualizes clusters using MDS and places annotations with class labels and centroids for each cluster.
        Params:
            X (array-like): Input data to be reduced using MDS.
            cluster_labels (array-like): Cluster labels for each data point.
            actual_labels (array-like): True class labels for the data.
            class_names (list): List of class names corresponding to each cluster label.
        Returns:
            None: Displays an MDS scatter plot of the clusters with annotated class labels.
    """
    mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean')
    X_reduced = mds.fit_transform(X)

    plt.figure(figsize=(18, 9))
    cmap = plt.get_cmap('viridis', len(np.unique(cluster_labels)))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap=cmap, edgecolor='k', s=60)

    unique_labels = np.unique(cluster_labels)
    centroids = {}
    cluster_to_class = {}

    for label in unique_labels:
        indices = np.where(cluster_labels == label)
        if len(indices[0]) > 0:
            most_common_label = np.bincount(actual_labels[indices]).argmax()
            cluster_to_class[label] = class_names[most_common_label]
            centroids[label] = np.mean(X_reduced[indices], axis=0)
        else:
            cluster_to_class[label] = "Unknown"

    for label, centroid in centroids.items():
        plt.text(centroid[0], centroid[1], cluster_to_class[label], fontsize=9, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    handles = [plt.Line2D([0], [0], marker='o', color=cmap(i / max(cluster_labels)), linestyle='', markersize=10,
                          markeredgecolor='k') for i in unique_labels]
    labels = [f"{cluster_to_class[label]} ({np.count_nonzero(cluster_labels == label)})" for label in unique_labels]
    plt.legend(handles, labels, title='Classes', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    plt.title('MDS Visualization of Clusters')
    plt.xlabel('MDS Component 1')
    plt.ylabel('MDS Component 2')
    plt.tight_layout()
    plt.show()

# Isomap Visualization
def visualize_clusters_isomap(X, cluster_labels, actual_labels, class_names):
    """
        Visualizes clusters using Isomap and places annotations with class labels and centroids for each cluster.
        Params:
            X (array-like): Input data to be reduced using Isomap.
            cluster_labels (array-like): Cluster labels for each data point.
            actual_labels (array-like): True class labels for the data.
            class_names (list): List of class names corresponding to each cluster label.
        Returns:
            None: Displays an Isomap scatter plot of the clusters with annotated class labels.
    """
    isomap = Isomap(n_neighbors=10, n_components=2)
    X_reduced = isomap.fit_transform(X)

    plt.figure(figsize=(18, 9))
    cmap = plt.get_cmap('viridis', len(np.unique(cluster_labels)))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap=cmap, edgecolor='k', s=60)

    unique_labels = np.unique(cluster_labels)
    centroids = {}
    cluster_to_class = {}

    for label in unique_labels:
        indices = np.where(cluster_labels == label)
        if len(indices[0]) > 0:
            most_common_label = np.bincount(actual_labels[indices]).argmax()
            cluster_to_class[label] = class_names[most_common_label]
            centroids[label] = np.mean(X_reduced[indices], axis=0)
        else:
            cluster_to_class[label] = "Unknown"

    for label, centroid in centroids.items():
        plt.text(centroid[0], centroid[1], cluster_to_class[label], fontsize=9, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    handles = [plt.Line2D([0], [0], marker='o', color=cmap(i / max(cluster_labels)), linestyle='', markersize=10,
                          markeredgecolor='k') for i in unique_labels]
    labels = [f"{cluster_to_class[label]} ({np.count_nonzero(cluster_labels == label)})" for label in unique_labels]
    plt.legend(handles, labels, title='Classes', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    plt.title('Isomap Visualization of Clusters')
    plt.xlabel('Isomap Component 1')
    plt.ylabel('Isomap Component 2')
    plt.tight_layout()
    plt.show()

def visualize_clusters(X, cluster_labels, actual_labels, class_names, viz):
    if viz == 'tsne':
        visualize_clusters_tsne(X, cluster_labels, actual_labels, class_names)
    elif viz == 'pca':
        visualize_clusters_pca(X, cluster_labels, actual_labels, class_names)
    # elif viz == 'opentsne':
    #     visualize_clusters_opentsne(X, cluster_labels, actual_labels, class_names)
    elif viz == 'isomap':
        visualize_clusters_isomap(X, cluster_labels, actual_labels, class_names)
    elif viz == 'mds':
        visualize_clusters_mds(X, cluster_labels, actual_labels, class_names)
    else:
        raise ValueError("Visualization type not recognized.")