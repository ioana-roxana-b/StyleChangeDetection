import json

import pandas as pd

from test_scripts import test

def test_graph():
    # Define the paths for configuration files
    classifier_config_path = f'classification_configs/graph_classifier_config.json'

    data_path = f'graph_features_hviii.csv'
    data_df = pd.read_csv(data_path)

    # Load classifier configurations from JSON file
    with open(classifier_config_path, 'r') as f:
        classifiers_config = json.load(f)

    # Call the function to process the loaded JSON configuration
    test.process_classifier_config('all', classifiers_config, data_df, viz = 'tsne')