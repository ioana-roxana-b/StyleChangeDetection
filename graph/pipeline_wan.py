import json
import os

from text_preprocessing import text_preprocessing
from graph import preprocessing, graph_features

def pipeline_wan(text_name, file_path, label, language):
    output_path = f"Outputs/WANS/{text_name}"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    chunks = text_preprocessing.split_into_chapters(text=file_path, label=label)
    sentences = text_preprocessing.split_into_phrases(chunks)
    preprocessed_text = preprocessing.preprocessing(sentences, punctuations=False, stopwords=False, lemmatizer=False, language=language)
    wans = preprocessing.construct_wans(preprocessed_text, include_pos=True, output_dir=output_path)
    features = graph_features.extract_features(wans)
    graph_features.save_features_to_json(features, filename=f"{output_path}/{text_name}.json")
    graph_features.extract_lexical_syntactic_features(features, top_n=20, filename = f"{output_path}/{text_name}.csv")