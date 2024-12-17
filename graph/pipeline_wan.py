import json
from text_preprocessing import text_preprocessing
from graph import preprocessing, graph_features

def pipeline_wan():
    # file_path = "Corpus/Shakespeare/TNK.txt"
    # chunks = text_preprocessing.split_into_chapters(text=file_path, label="Shakespeare|Fletcher|DOS|TOL")
    #
    # # with open("acts.json", "w") as file:
    # #     json.dump(chunks, file, indent=4)
    #
    # sentences = text_preprocessing.split_into_phrases(chunks)
    #
    # # with open("sentence.json", "w") as file:
    # #     json.dump(sentences, file, indent=4)
    #
    # preprocessed_text = preprocessing.preprocessing(sentences, punctuations=True, stopwords=True, lemmatizer=True)
    #
    # # with open("preprocessed_acts.json", "w") as file:
    # #     json.dump(preprocessed_text, file, indent=4)
    #
    # wans = preprocessing.construct_wans(preprocessed_text, include_pos=True, output_dir="tnk_wans")
    #
    # #wans = preprocessing.load_wans("wans.json")
    #
    # features = graph_features.extract_features(wans)
    #
    # graph_features.save_features_to_json(features, filename="features_tnk.json")
    #
    # graph_features.extract_lexical_syntactic_features(features, top_n=10, filename = "graph_features_tnk.csv")
    #
    loaded_wan = preprocessing.load_wan("Fletcher ACT II. Scene 2.", input_dir="tnk_wans")
    if loaded_wan:
        preprocessing.plotly_visualize_wan(loaded_wan, "Fletcher ACT II. Scene 2.")

    loaded_wan_2 = preprocessing.load_wan("Shakespeare ACT I. Scene 2.", input_dir="tnk_wans")
    if loaded_wan_2:
        preprocessing.plotly_visualize_wan(loaded_wan_2, "Shakespeare ACT I. Scene 2.")
