from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as sk_tsne
import numpy as np
from openTSNE import TSNE
import matplotlib.pyplot as plt

def visualize_clusters(X, labels, class_names):
    """
       Visualizes clusters using a scatter plot with annotated class labels.
       Params:
           X (array-like): 2D input data to be visualized.
           labels (array-like): Cluster labels for each data point.
           class_names (list): List of class names corresponding to each cluster label.
       Returns:
           None: Displays a scatter plot of the clusters.
       """

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
    """
        Visualizes clusters using t-SNE for dimensionality reduction and custom annotations for clusters.
        Params:
            X (array-like): Input data to be reduced using t-SNE.
            cluster_labels (array-like): Cluster labels for each data point.
            actual_labels (array-like): True class labels for the data.
            class_names (list): List of class names corresponding to each cluster label.
        Returns:
            None: Displays a t-SNE scatter plot of the clusters with enhanced visualization.
    """
    # Apply t-SNE reduction
    tsne = sk_tsne(n_components=2, random_state=42, perplexity=50)
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
    plt.xticks(ticks=np.arange(min(X_reduced[:, 0]), max(X_reduced[:, 0]) + 1, 10))
    plt.yticks(ticks=np.arange(min(X_reduced[:, 1]), max(X_reduced[:, 1]) + 1, 10))

    # Reduce grid line visibility or remove grid
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Lighter grid

    # Legend outside the plot
    handles = [plt.Line2D([0], [0], marker='o', color=cmap(i / max(cluster_labels)), linestyle='', markersize=10,
                          markeredgecolor='k') for i in unique_labels]
    labels = [f"{cluster_to_class[label]} ({np.count_nonzero(cluster_labels == label)})" for label in unique_labels]
    plt.legend(handles, labels, title='Classes', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    plt.title('Enhanced t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.show()


def visualize_clusters_tsne_label(X, cluster_labels, actual_labels, class_names):
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
    tsne = sk_tsne(n_components=2, random_state=42, perplexity=100)
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
    plt.xticks(ticks=np.arange(min(X_reduced[:, 0]), max(X_reduced[:, 0]) + 1, 10))
    plt.yticks(ticks=np.arange(min(X_reduced[:, 1]), max(X_reduced[:, 1]) + 1, 10))

    # Reduce grid line visibility or remove grid
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Lighter grid

    # Legend outside the plot
    handles = [plt.Line2D([0], [0], marker='o', color=cmap(i / max(cluster_labels)), linestyle='', markersize=10,
                          markeredgecolor='k') for i in unique_labels]
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
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', marker='o', edgecolor='k', s=50)

    # Creating custom legend with manual mapping
    unique_clusters = np.unique(cluster_labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / max(unique_clusters)),
                          markersize=10, markeredgecolor='k') for i in unique_clusters]
    new_labels = [cluster_to_label_mapping[cluster] for cluster in unique_clusters]
    plt.legend(handles, new_labels, title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title('Corrected PCA Clusters Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True)
    plt.show()

def visualize_clusters_opentsne_label(X, cluster_labels, actual_labels, class_names):
    """
        Visualizes clusters using OpenTSNE and places annotations with class labels and centroids for each cluster.
        Params:
            X (array-like): Input data to be reduced using OpenTSNE.
            cluster_labels (array-like): Cluster labels for each data point.
            actual_labels (array-like): True class labels for the data.
            class_names (list): List of class names corresponding to each cluster label.
        Returns:
            None: Displays an OpenTSNE scatter plot of the clusters with annotated class labels.
    """
    # Apply OpenTSNE reduction
    tsne = TSNE(n_jobs=-1, perplexity=50, random_state=42)
    X_reduced = tsne.fit(X)

    plt.figure(figsize=(18, 9))  # Adjust the figure size as necessary
    cmap = plt.get_cmap('viridis', len(np.unique(cluster_labels)))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap=cmap, edgecolor='k', s=60)

    unique_labels = np.unique(cluster_labels)
    cluster_to_class = {}
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

    # Customize the ticks to make grid less dense
    plt.xticks(ticks=np.linspace(np.min(X_reduced[:, 0]), np.max(X_reduced[:, 0]), num=10))
    plt.yticks(ticks=np.linspace(np.min(X_reduced[:, 1]), np.max(X_reduced[:, 1]), num=10))
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)  # Lighter grid

    # Legend outside the plot
    handles = [
        plt.Line2D(
            [0], [0], marker='o', color=cmap(i / max(cluster_labels)), linestyle='', markersize=10,
            markeredgecolor='k'
        ) for i in unique_labels
    ]
    labels = [f"{cluster_to_class[label]} ({np.count_nonzero(cluster_labels == label)})" for label in unique_labels]
    plt.legend(handles, labels, title='Classes', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Place text annotations
    for label, centroid in centroids.items():
        plt.text(
            centroid[0], centroid[1], cluster_to_class[label], fontsize=9, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5')
        )

    plt.title('OpenTSNE Visualization of Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.show()
