import pandas as pd
from tods.tods import schemas as schemas_utils
from tods.tods import generate_dataset, evaluate_pipeline

table_path = '../Outputs/sentence_features.csv'
target_index = 4 # what column is the target
metric = 'F1_MACRO' # F1 on both label 0 and 1

# Read data and generate dataset
df = pd.read_csv(table_path)
dataset = generate_dataset(df, target_index)

# Load the default pipeline
pipeline = schemas_utils.load_default_pipeline()

# Run the pipeline
pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
print(pipeline_result)