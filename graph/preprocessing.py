import os
import pickle
import re

from matplotlib import pyplot as plt
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import os
import networkx as nx
import pickle
from pyvis.network import Network

def process_sentence(sentence, stop_words=None, lemmatizer_instance=None, remove_punctuations=False):
    """
    Process a single sentence by applying common text preprocessing steps.
    Params:
        sentence (str): The input sentence to be processed.
        stop_words (set): A set of stopwords to be removed (if applicable).
        lemmatizer_instance (object): An instance of a lemmatizer (e.g., WordNetLemmatizer).
        remove_punctuations (bool): If True, removes all punctuations from the sentence.

    Returns:
        str: The processed sentence as a single string.
    """
    # Convert to lowercase
    sentence = sentence.lower()

    # Tokenize the sentence
    tokens = word_tokenize(sentence)

    # Remove stopwords
    if stop_words:
        tokens = [word for word in tokens if word not in stop_words]

    # Remove punctuations
    if remove_punctuations:
        tokens = [re.sub(r'[^\w\s]', '', word) for word in tokens]
        tokens = [word for word in tokens if word.strip() != '']

    # Apply lemmatization
    if lemmatizer_instance:
        tokens = [lemmatizer_instance.lemmatize(word) for word in tokens]

    # Reconstruct the processed sentence
    return ' '.join(tokens)


def preprocessing(text=None, stopwords=False, lemmatizer=False, punctuations=False):
    """
    Preprocess text data.
    Params:
        text (dict): Dictionary where values can be strings or lists of sentences
        stopwords (bool): Remove stopwords if True
        lemmatizer (bool): Apply lemmatization if True
        punctuations (bool): Remove punctuations if True

    Returns:
        dict: Preprocessed text in the same format as input
    """
    if not text:
        raise ValueError("Input text cannot be None.")

    # Initialize optional components
    stop_words = set(nltk_stopwords.words('english')) if stopwords else None
    lemmatizer_instance = WordNetLemmatizer() if lemmatizer else None

    processed_text = {}

    for key, value in text.items():
        if isinstance(value, str):
            # Process a single string
            processed_text[key] = process_sentence(
                value, stop_words, lemmatizer_instance, punctuations
            )
        elif isinstance(value, list):
            # Process a list of sentences
            processed_text[key] = [
                process_sentence(sentence, stop_words, lemmatizer_instance, punctuations)
                for sentence in value
            ]
        else:
            raise TypeError(f"Expected str or list for key '{key}', got '{type(value).__name__}'.")

    return processed_text


def construct_wans(text=None, output_dir="wans"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wans = {}

    # Iterate through scenes in the text
    for scene in text:
        wan = nx.Graph()

        # Iterate through sentences in the scene
        for sentence in text[scene]:
            words = sentence.split()

            # Add edges only between adjacent words
            for i in range(len(words) - 1):
                word1 = words[i]
                word2 = words[i + 1]
                wan.add_node(word1)
                wan.add_node(word2)
                if wan.has_edge(word1, word2):
                    wan[word1][word2]['weight'] += 1
                else:
                    wan.add_edge(word1, word2, weight=1)

        # Store WAN in the dictionary
        wans[scene] = wan

        # Save each WAN as a pickle file
        with open(f"{output_dir}/{scene}.pkl", "wb") as f:
            pickle.dump(wan, f)

    print(f"WANs saved in {output_dir}")
    return wans


def load_wan(scene, input_dir="wans"):
    filepath = f"{input_dir}/{scene}.pkl"
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        print(f"WAN for scene '{scene}' not found.")
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



def plotly_visualize_wan(wan, scene_name):
    """
    Create an interactive WAN visualization using Plotly.

    Parameters:
        wan (networkx.Graph): The word association network to visualize.
        scene_name (str): The name of the scene for labeling.
    """
    pos = nx.spring_layout(wan, seed=42)

    # Extract node and edge positions
    node_x = [pos[node][0] for node in wan.nodes]
    node_y = [pos[node][1] for node in wan.nodes]
    edge_x = []
    edge_y = []
    edge_weights = []
    edge_text = []

    for edge in wan.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_weights.append(edge[2]['weight'])
        edge_text.append(f"{edge[0]} ↔ {edge[1]} (Weight: {edge[2]['weight']})")

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines',
        text=edge_text
    )

    # Create node trace with custom hover information
    node_adjacencies = []
    node_text = []
    for node in wan.nodes:
        adjacencies = list(wan[node])
        node_adjacencies.append(len(adjacencies))
        connected_nodes = ', '.join(adjacencies)
        node_text.append(f"{node}<br>Connected to: {connected_nodes}")

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
        hovertext=node_text
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