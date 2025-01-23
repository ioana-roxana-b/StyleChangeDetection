import os
import json
from text_preprocessing import text_preprocessing
from graph import preprocessing, graph_features

def process_problem(problem_id, data_dir, truth_lookup, output_dir):
    if problem_id not in truth_lookup:
        return None

    truth_label = truth_lookup[problem_id]
    problem_folder = os.path.join(data_dir, problem_id)

    # Get sorted list of valid document files
    doc_files = sorted([f for f in os.listdir(problem_folder) if f.endswith('.txt')])

    # Check for insufficient documents
    if len(doc_files) < 2:
        print(f"Insufficient documents for problem {problem_id}. Found {len(doc_files)} files.")
        return None

    # Read documents
    known_documents = []
    for doc_file in doc_files[:-1]:  # All except the last are known
        doc_path = os.path.join(problem_folder, doc_file)
        with open(doc_path, 'r', encoding='utf-8-sig') as f:
            known_documents.append(f.read())

    # The last document is the questioned one
    question_document = doc_files[-1]
    with open(os.path.join(problem_folder, question_document), 'r', encoding='utf-8-sig') as f:
        question_document = f.read()

    # Prepare documents dictionary
    documents_dict = {f"{truth_label}_{problem_id}_known_doc_{idx + 1}": doc for idx, doc in enumerate(known_documents)}
    documents_dict[f"{truth_label}_{problem_id}_question_doc"] = question_document

    # Preprocess and feature extraction (unchanged)
    sentences = text_preprocessing.split_into_phrases(documents_dict)
    preprocessed_text = preprocessing.preprocessing(sentences, punctuations=False, stopwords=True, lemmatizer=True, language="en")

    # Construct WANS and extract features
    wans = preprocessing.construct_wans(
        preprocessed_text,
        include_pos=True,
        output_dir=os.path.join(output_dir, f"wan_{problem_id}")
    )
    features = graph_features.extract_features(wans)

    # Add truth label
    for scene in features:
        features[scene]['label'] = truth_label

    # Save features
    graph_features.save_features_to_json(features, filename=os.path.join(output_dir, f"features_{problem_id}.json"))
    graph_features.extract_lexical_syntactic_features(
        features, top_n=10, filename=os.path.join(output_dir, f"graph_features_{problem_id}.csv")
    )

    return problem_id, truth_label


def load_truth_file(truth_file):
    """Parses the truth.txt file into a dictionary."""
    truth_lookup = {}
    with open(truth_file, 'r') as f:
        for line in f:
            problem_id, label = line.strip().split()
            truth_lookup[problem_id] = label
    return truth_lookup

def pipeline_pan(train_dir, test_dir, train_truth_file, test_truth_file, output_train_dir, output_test_dir):
    # Load ground truth data for training and testing
    train_truth_lookup = load_truth_file(train_truth_file)
    test_truth_lookup = load_truth_file(test_truth_file)

    # Initialize results
    train_results = []
    test_results = []

    # Process training data
    for problem_id in os.listdir(train_dir):
        result = process_problem(problem_id, train_dir, train_truth_lookup, output_train_dir)
        if result:
            train_results.append(result)

    # Process testing data
    for problem_id in os.listdir(test_dir):
        result = process_problem(problem_id, test_dir, test_truth_lookup, output_test_dir)
        if result:
            test_results.append(result)

    print("Pipeline completed successfully!")
    return train_results, test_results
