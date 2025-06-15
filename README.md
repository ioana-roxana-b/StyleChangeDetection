# StyleChangeDetection

The goal of this project is to develop and evaluate methods for detecting style changes in texts. This can be useful in authorship attribution, plagiarism detection, or analyzing narrative structures in literature. The repository includes scripts for preprocessing text, extracting features, training models, and visualizing results.

## Repository Structure

- **main.py**: The primary entry point for running the style change detection pipeline.
- **parse_args.py**: Utility for handling command-line arguments.
- **requirements.txt**: Lists the dependencies required to run the project.
- **Corpus/**: Contains textual data used for training and testing, including works by various authors like Charles Dickens, Dostoevsky, Shakespeare, and Tolstoy.
- **classification/**: Scripts for classification tasks, including specific implementations for different datasets or challenges (e.g., PAN).
- **classification_configs/**: Configuration files for different classifiers used in the project.
- **features_methods/**: Modules for feature engineering and extraction, including TF-IDF based features.
- **feature_configs/**: Configuration files for feature extraction at different levels (chapter, sentence).
- **graph/**: Tools for graph-based analysis and features, potentially for visualizing or modeling text relationships.
- **models/**: Contains implementations of supervised and unsupervised models for style change detection, along with visualization tools.
- **text_preprocessing/**: Scripts for cleaning and preparing text data for analysis.
- **Outputs/**: Directory for storing results and outputs from the models.
- **test_scripts/**: Scripts for testing the implemented methods on various datasets.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

### Usage

To run the style change detection pipeline, use the `main.py` script. You can specify various parameters through command-line arguments as defined in `parse_args.py`. Here's a basic example of how to run the project:

```bash
python main.py --config classification_configs/classifiers_config.json
```

Refer to the specific configuration files in `classification_configs/` and `feature_configs/` for customizing the pipeline to different datasets or analysis requirements.

