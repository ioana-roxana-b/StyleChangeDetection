import json
from text_preprocessing import text_preprocessing
from graph import preprocessing

def pipeline_wan():
    file_path = "Corpus/Shakespeare/TNK.txt"
    chunks = text_preprocessing.split_into_chapters(text=file_path, label="Shakespeare|Fletcher")

    with open("acts.json", "w") as file:
        json.dump(chunks, file, indent=4)

    sentences = text_preprocessing.split_into_phrases(chunks)

    with open("sentence.json", "w") as file:
        json.dump(sentences, file, indent=4)

    preprocessed_text = preprocessing.preprocessing(sentences, punctuations=True)

    with open("preprocessed_acts.json", "w") as file:
        json.dump(preprocessed_text, file, indent=4)

    preprocessing.construct_wans(preprocessed_text)

    loaded_wan = preprocessing.load_wan("Shakespeare ACT I. Scene 1.")
    if loaded_wan:
        preprocessing.plotly_visualize_wan(loaded_wan, "Shakespeare ACT I. Scene 1.")
