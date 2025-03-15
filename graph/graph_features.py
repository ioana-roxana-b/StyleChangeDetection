import csv
import json
import os
from collections import Counter

import networkx as nx
import pandas as pd


def extract_features(wans):
    features = {}

    for scene, wan in wans.items():
        scene_features = {}

        # Graph-Level Metrics
        scene_features['average_degree'] = sum(dict(wan.degree()).values()) / wan.number_of_nodes()
        scene_features['density'] = nx.density(wan)
        scene_features['average_clustering'] = nx.average_clustering(wan.to_undirected())
        scene_features['assortativity'] = nx.degree_assortativity_coefficient(wan)

        # Node-Level Metrics
        scene_features['degree_centrality'] = nx.degree_centrality(wan)
        scene_features['in_degree'] = dict(wan.in_degree())
        scene_features['out_degree'] = dict(wan.out_degree())
        scene_features['betweenness_centrality'] = nx.betweenness_centrality(wan)
        scene_features['closeness_centrality'] = nx.closeness_centrality(wan)
        scene_features['eigenvector_centrality'] = nx.eigenvector_centrality(wan, max_iter=50000)

        # Edge-Level Metrics
        edge_weights = {edge: data['weight'] for edge, data in wan.edges.items()}
        scene_features['edge_weights'] = edge_weights

        features[scene] = scene_features

    return features


def convert_keys_to_str(data):
    if isinstance(data, dict):
        return {str(key): convert_keys_to_str(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_keys_to_str(element) for element in data]
    else:
        return data

def save_features_to_json(features, filename="features_en.json"):
    # Convert all keys to strings to avoid JSON encoding errors
    cleaned_features = convert_keys_to_str(features)

    with open(filename, "w") as f:
        json.dump(cleaned_features, f, indent=4)
    #print(f"Features saved to {filename}")

def extract_lexical_syntactic_features(features, top_n=10, filename="graph_features.csv"):
    feature_vectors = {}

    for scene, scene_features in features.items():
        vector = {}

        # Degree Centrality: Extract Top-N Collocations
        degree_top_words = sorted(scene_features['degree_centrality'].items(),
                                  key=lambda x: x[1], reverse=True)[:top_n]
        for i, (_, value) in enumerate(degree_top_words, start=1):
            vector[f'degree_centrality_{i}'] = value

        # Closeness Centrality: Extract Top-N Chunk Phrases
        closeness_top_words = sorted(scene_features['closeness_centrality'].items(),
                                     key=lambda x: x[1], reverse=True)[:top_n]
        for i, (_, value) in enumerate(closeness_top_words, start=1):
            vector[f'closeness_centrality_{i}'] = value

        # Betweenness Centrality: Extract Bigrams/Trigrams
        betweenness_top_words = sorted(scene_features['betweenness_centrality'].items(),
                                       key=lambda x: x[1], reverse=True)[:top_n]
        for i, (_, value) in enumerate(betweenness_top_words, start=1):
            vector[f'betweenness_centrality_{i}'] = value

        # Eigenvector Centrality: Extract Relevant Words
        eigenvector_top_words = sorted(scene_features['eigenvector_centrality'].items(),
                                       key=lambda x: x[1], reverse=True)[:top_n]
        for i, (_, value) in enumerate(eigenvector_top_words, start=1):
            vector[f'eigenvector_centrality_{i}'] = value

        # Include Graph-Level Metrics
        vector['average_degree'] = scene_features['average_degree']
        vector['density'] = scene_features['density']
        vector['average_clustering'] = scene_features['average_clustering']
        vector['assortativity'] = scene_features['assortativity']

        # Include Edge-Level Metrics (optional)
        edge_weights = list(scene_features['edge_weights'].values())
        vector['edge_weights_mean'] = sum(edge_weights) / len(edge_weights) if edge_weights else 0

        feature_vectors[scene] = vector

    # Create a DataFrame from the feature vectors and save it to CSV
    df = pd.DataFrame.from_dict(feature_vectors, orient='index').fillna(0)
    df.index.name = 'label'
    df.to_csv(filename, index=True)

    return feature_vectors

