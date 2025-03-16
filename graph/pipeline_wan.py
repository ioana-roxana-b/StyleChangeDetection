import json
import os

from text_preprocessing import text_preprocessing
from graph import preprocessing, graph_features

def pipeline_wan(text_name, file_path, label, language, wan_config):
    output_path = f"Outputs/WANS/{text_name}"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open("wan_configs/wan_configs.json", 'r') as f:
        data = json.load(f)

    config = data.get(wan_config, {})

    if not isinstance(config, dict):
        raise ValueError(f"The key '{wan_config}' does not point to a valid dictionary in the configuration file.")

    punctuations = config['punctuations']
    stopwords = config['stopwords']
    lemmatizer = config['lemmatizer']
    include_pos = config['include_pos']

    chunks = text_preprocessing.split_into_chapters(text=file_path, label=label)
    sentences = text_preprocessing.split_into_phrases(chunks)
    preprocessed_text = preprocessing.preprocessing(sentences, punctuations=punctuations, stopwords=stopwords, lemmatizer=lemmatizer, language=language)
    wans = preprocessing.construct_wans(preprocessed_text, include_pos=include_pos, output_dir=output_path)
    features = graph_features.extract_features(wans)
    graph_features.save_features_to_json(features, filename=f"{output_path}/{text_name}.json")
    graph_features.extract_lexical_syntactic_features(features, top_n=20, filename = f"{output_path}/{text_name}.csv")