import numpy as np

import features
import tf_idf_features

def save_features(feature_specs = None):
    """
    Extracts features based on the given specifications and aggregates them into a dictionary.
    Params:
        feature_specs (dict): Dictionary containing feature function names as keys and parameters as values.
                              Example: {"feature_func_name": {"param1": value1, "param2": value2}}
    Returns:
        dict: A nested dictionary containing all the extracted features for each text element.
    """

    all_features = {}

    # Iterate over each feature specification
    for feature_name, params in feature_specs.items():
        # Try to locate the feature function in the current namespace
        feature_func = globals().get(feature_name)

        # If not found, look for the feature in the `features` module
        if feature_func is None:
            feature_func = getattr(features, feature_name, None)

        # If still not found, look for the feature in the `tf_idf_features` module
        if feature_func is None:
            feature_func = getattr(tf_idf_features, feature_name, None)

        if feature_func:
            # Handle sentence-level features separately
            if feature_name.startswith("sentence"):
                # Call the function with specified parameters
                feature_vector = feature_func(**params)

                # Iterate over each key (e.g., chapter/segment) in the returned dictionary
                for key in feature_vector.keys():
                    # Iterate over each sub-key (e.g., sentence) and its value
                    for value in feature_vector[key].keys():
                        value_to_add = feature_vector[key][value]
                        if not isinstance(value_to_add, list):
                            # Ensure value_to_add is a list (for consistent aggregation)
                            value_to_add = [value_to_add]

                        # Append values to the corresponding tuple (key, value) in `all_features`
                        all_features.setdefault((key, value), []).extend(value_to_add)

            else:
                # Handle other feature functions that do not operate at the sentence level
                feature_vector = feature_func(**params if params else {})

                # Iterate over each key-value pair in the returned dictionary
                for key, value in feature_vector.items():
                    # Convert NumPy arrays to lists for compatibility
                    if isinstance(value, np.ndarray):
                        value = value.tolist()

                    if key in all_features:
                        # If the key already exists, merge the new values
                        if not isinstance(all_features[key], list):
                            # Ensure the existing feature is in list format
                            all_features[key] = [all_features[key]]

                        if isinstance(value, list):
                            # Extend the existing list with new values
                            all_features[key].extend(value)

                        else:
                            all_features[key].append(value)

                    else:
                        # If the key is new, directly add it to `all_features`
                        all_features[key] = value

        else:
            print(f"Feature function {feature_name} not found.")
            continue

    return all_features
