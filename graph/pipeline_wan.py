import json
from text_preprocessing import text_preprocessing
from graph import preprocessing, graph_features

def pipeline_wan():
    file_path = "Corpus/Combined_texts/Crime_Anna_rus.txt"
    chunks = text_preprocessing.split_into_chapters(text=file_path, label="Shakespeare|Fletcher|DOS|TOL")

    # with open("acts.json", "w") as file:
    #     json.dump(chunks, file, indent=4)

    sentences = text_preprocessing.split_into_phrases(chunks)

    # with open("sentence.json", "w") as file:
    #     json.dump(sentences, file, indent=4)

    preprocessed_text = preprocessing.preprocessing(sentences, punctuations=True, stopwords=True, lemmatizer=True, language='ru')

    # with open("preprocessed_acts.json", "w") as file:
    #     json.dump(preprocessed_text, file, indent=4)

    wans = preprocessing.construct_wans(preprocessed_text, include_pos=True, output_dir="dos_wans_rus_new")

    #wans = preprocessing.load_wans("wans.json")

    features = graph_features.extract_features(wans)

    graph_features.save_features_to_json(features, filename="features_rus.json")

    graph_features.extract_lexical_syntactic_features(features, top_n=20, filename = "graph_features_rus.csv")

    # fletcher_wan = preprocessing.load_wan("Fletcher ACT II. Scene 2.", input_dir="tnk_wans")
    # if fletcher_wan:
    #     preprocessing.plotly_visualize_wan(fletcher_wan, "Fletcher ACT II. Scene 2.")
    #
    # shakespeare_wan = preprocessing.load_wan("Shakespeare ACT I. Scene 2.", input_dir="tnk_wans")
    # if shakespeare_wan:
    #     preprocessing.plotly_visualize_wan(shakespeare_wan, "Shakespeare ACT I. Scene 1.")
    #
    # dos_wan = preprocessing.load_wan("DOS CHAPTER I PART I", input_dir="dos_wans")
    # if dos_wan:
    #     preprocessing.plotly_visualize_wan(dos_wan, "DOS CHAPTER I PART I")
    #
    # tol_wan = preprocessing.load_wan("TOL CHAPTER 1 PART ONE", input_dir="dos_wans")
    # if tol_wan:
    #     preprocessing.plotly_visualize_wan(tol_wan, "TOL CHAPTER 1 PART ONE")

    dos_wan_rus = preprocessing.load_wan("DOS CHAPTER 1 I", input_dir="dos_wans_rus")
    if dos_wan_rus:
        preprocessing.plotly_visualize_wan(dos_wan_rus, "DOS CHAPTER 1 I")

    tol_wan_rus = preprocessing.load_wan("TOL CHAPTER 1 PART ONE", input_dir="dos_wans_rus")
    if tol_wan_rus:
        preprocessing.plotly_visualize_wan(tol_wan_rus, "TOL CHAPTER 1 PART ONE")

    dos_wan_rus = preprocessing.load_wan("DOS CHAPTER 1 I", input_dir="dos_wans_rus_new")
    if dos_wan_rus:
        preprocessing.plotly_visualize_wan(dos_wan_rus, "DOS CHAPTER 1 I")

    tol_wan_rus = preprocessing.load_wan("TOL CHAPTER 1 PART ONE", input_dir="dos_wans_rus_new")
    if tol_wan_rus:
        preprocessing.plotly_visualize_wan(tol_wan_rus, "TOL CHAPTER 1 PART ONE")