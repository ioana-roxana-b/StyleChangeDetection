import json
import re
from matplotlib import pyplot as plt
from nltk import pos_tag
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import os
import networkx as nx
import pickle
from pyvis.network import Network
import spacy
from spacy.lang.ru.examples import sentences
import stopwordsiso as stopwords

# Load Russian stopwords using stopwords-iso
stopwords_ru = stopwords.stopwords("ru")

# Load SpaCy model for Russian
nlp_ru = spacy.load("ru_core_news_md")


def process_sentence(
    sentence, stop_words=None, lemmatizer_instance=None, remove_punctuations=False, language="en"
):
    """
    Process a single sentence by applying common text preprocessing steps.
    Params:
        sentence (str): The input sentence to be processed.
        stop_words (set): A set of stopwords to be removed (if applicable).
        lemmatizer_instance (object): An instance of a lemmatizer (e.g., WordNetLemmatizer for English).
        remove_punctuations (bool): If True, removes all punctuations from the sentence.
        language (str): Language of the text, either 'en' (English) or 'ru' (Russian).

    Returns:
        str: The processed sentence as a single string.
    """
    # Convert to lowercase
    sentence = sentence.lower()

    # Tokenize and process based on the language
    if language == "en":
        # Tokenize using NLTK for English
        tokens = word_tokenize(sentence)

        # Remove stopwords
        if stop_words:
            tokens = [word for word in tokens if word not in stop_words]

        # Remove punctuations
        if remove_punctuations:
            tokens = [re.sub(r"[^\w\s]", "", word) for word in tokens]
            tokens = [word for word in tokens if word.strip() != ""]

        # Apply lemmatization
        if lemmatizer_instance:
            tokens = [lemmatizer_instance.lemmatize(word) for word in tokens]

    elif language == "ru":
        # Tokenize and lemmatize using SpaCy for Russian
        doc = nlp_ru(sentence)
        tokens = [token.lemma_ for token in doc if not token.is_punct]

        # Remove stopwords
        if stop_words:
            tokens = [word for word in tokens if word not in stop_words]

    else:
        raise ValueError("Unsupported language. Use 'en' for English or 'ru' for Russian.")

    # Reconstruct the processed sentence
    return " ".join(tokens)


def preprocessing(
    text=None, stopwords=False, lemmatizer=False, punctuations=False, language="en"
):
    """
    Preprocess text data.
    Params:
        text (dict): Dictionary where values can be strings or lists of sentences.
        stopwords (bool): Remove stopwords if True.
        lemmatizer (bool): Apply lemmatization if True.
        punctuations (bool): Remove punctuations if True.
        language (str): Language of the text, either 'en' (English) or 'ru' (Russian).

    Returns:
        dict: Preprocessed text in the same format as input.
    """
    if not text:
        raise ValueError("Input text cannot be None.")

    # Initialize optional components based on the language
    if language == "en":
        stop_words = set(nltk_stopwords.words("english")) if stopwords else None
        lemmatizer_instance = WordNetLemmatizer() if lemmatizer else None
    elif language == "ru":
        stop_words = stopwords_ru if stopwords else None
        lemmatizer_instance = None  # SpaCy handles lemmatization for Russian
    else:
        raise ValueError("Unsupported language. Use 'en' for English or 'ru' for Russian.")

    processed_text = {}

    for key, value in text.items():
        if isinstance(value, str):
            # Process a single string
            processed_text[key] = process_sentence(
                value, stop_words, lemmatizer_instance, punctuations, language
            )
        elif isinstance(value, list):
            # Process a list of sentences
            processed_text[key] = [
                process_sentence(sentence, stop_words, lemmatizer_instance, punctuations, language)
                for sentence in value
            ]
        else:
            raise TypeError(f"Expected str or list for key '{key}', got '{type(value).__name__}'.")

    return processed_text


def construct_wans(text=None, output_dir="dos_wans", include_pos=False):
    if text is None or not isinstance(text, dict):
        raise ValueError("Input 'text' must be a dictionary with scenes as keys.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wans = {}

    for scene, sentences in text.items():
        wan = nx.DiGraph()

        for sentence in sentences:
            words = word_tokenize(sentence)

            # Add edges between adjacent words
            for i in range(len(words) - 1):
                word1, word2 = words[i].lower(), words[i + 1].lower()
                if wan.has_edge(word1, word2):
                    wan[word1][word2]['weight'] += 1
                else:
                    wan.add_edge(word1, word2, weight=1)

            # Add POS tagging if enabled
            if include_pos:
                pos_tags = pos_tag(words)
                for word, pos in pos_tags:
                    if word.lower() in wan.nodes:
                        wan.nodes[word.lower()]['pos'] = pos

        wan.remove_edges_from(nx.selfloop_edges(wan))
        wans[scene] = wan

        # Save WAN as a pickle file
        try:
            with open(f"{output_dir}/{scene}.pkl", "wb") as f:
                pickle.dump(wan, f)
        except IOError as e:
            print(f"Error saving {scene}: {e}")

    try:
        json_wans = {scene: nx.node_link_data(wan) for scene, wan in wans.items()}
        with open(f"{output_dir}.json", "w") as f:
            json.dump(json_wans, f, indent=4)
    except IOError as e:
        print(f"Error saving WANs as JSON: {e}")

    #print(f"WANs saved in {output_dir}")
    return wans

def plotly_visualize_wan(wan, scene_name, max_hover_connections=10):
    """
    Create an interactive WAN visualization using Plotly with directional edges and hover limits.

    Parameters:
        wan (networkx.DiGraph): The word association network to visualize.
        scene_name (str): The name of the scene for labeling.
        max_hover_connections (int): Maximum number of connections to display in hover text.
    """
    pos = nx.spring_layout(wan, seed=42, k=0.4)  # Spread nodes further apart

    # Extract node and edge positions
    node_x = [pos[node][0] for node in wan.nodes]
    node_y = [pos[node][1] for node in wan.nodes]

    edge_x = []
    edge_y = []
    edge_text = []

    for edge in wan.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(f"{edge[0]} → {edge[1]} (Weight: {edge[2]['weight']})")

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines',
        text=edge_text
    )

    # Create node trace with detailed hover information
    node_adjacencies = []
    node_text = []

    for node in wan.nodes:
        out_edges = list(wan.successors(node))  # Nodes this node points to
        in_edges = list(wan.predecessors(node))  # Nodes pointing to this node

        pos_tag = wan.nodes[node].get('pos', 'Unknown')

        out_connections = ', '.join(out_edges[:max_hover_connections])
        in_connections = ', '.join(in_edges[:max_hover_connections])

        out_more = len(out_edges) - max_hover_connections if len(out_edges) > max_hover_connections else 0
        in_more = len(in_edges) - max_hover_connections if len(in_edges) > max_hover_connections else 0

        out_text = f"Followed by: {out_connections} and {out_more} more..." if out_more else f"Followed by: {out_connections}"
        in_text = f"Follows: {in_connections} and {in_more} more..." if in_more else f"Follows: {in_connections}"

        node_adjacencies.append(len(out_edges) + len(in_edges))
        node_text.append(f"<b>{node}</b><br>POS: {pos_tag}<br>{out_text}<br>{in_text}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(node) for node in wan.nodes],
        hoverinfo='text',
        marker=dict(
            size=10,
            color=node_adjacencies,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Node Degree")
        ),
        textposition='top center',
        hovertext=node_text  # Use the formatted hover text here
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"WAN for Scene: {scene_name}",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    fig.show()

def load_wan(scene, input_dir="dos_wans"):
    filepath = f"{input_dir}/{scene}.pkl"
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        print(f"WAN for scene '{scene}' not found.")
        return None


def load_wans(json_file="wans.json"):
    try:
        with open(json_file, "r") as f:
            json_wans = json.load(f)

        # Convert back to NetworkX graphs
        wans = {scene: nx.node_link_graph(data) for scene, data in json_wans.items()}
        print("WANs successfully loaded.")
        return wans
    except IOError as e:
        print(f"Error loading WANs from JSON: {e}")
        return None
def visualize_wan(wan, scene_name):

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(wan, seed=42)
    nx.draw(wan, pos, with_labels=True, node_size=500, font_size=10, font_color="white")
    edge_labels = nx.get_edge_attributes(wan, 'weight')
    nx.draw_networkx_edge_labels(wan, pos, edge_labels=edge_labels)
    plt.title(f"WAN for Scene: {scene_name}")
    plt.show()


def create_interactive_wan(wan, output_path="interactive_wan.html"):
    """
    Create an interactive visualization of the WAN using pyvis with constrained physics.

    Parameters:
        wan (networkx.Graph): The word association network to visualize.
        output_path (str): Path to save the interactive HTML visualization.
    """
    net = Network(height="750px", width="100%", notebook=True)

    # Add nodes and edges
    for node in wan.nodes:
        net.add_node(node, title=node)

    for edge in wan.edges(data=True):
        net.add_edge(edge[0], edge[1], title=f"Weight: {edge[2]['weight']}", value=edge[2]['weight'])

    # Configure physics options for better stabilization
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {
          "enabled": true,
          "iterations": 1000
        }
      }
    }
    """)

    # Generate and save the interactive visualization
    net.show(output_path, notebook=False)

    print(f"Interactive WAN saved to {output_path}")


