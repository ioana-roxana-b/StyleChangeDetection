from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

def kmeans(X, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    silhouette_vals = silhouette_score(X, clusters, metric='euclidean')

    return clusters, silhouette_vals

def pca_dimension_reduction(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    return X_reduced

def isolation_forest(X):
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    anomalies = iso_forest.fit_predict(X)
    anomaly_mask = anomalies == -1

    return anomaly_mask


