import numpy as np

from features_methods import features
from features_methods import tf_idf_features


def save_features(feature_specs=None):
    """
    Extracts features based on the given specifications and aggregates them into a dictionary.
    Params:
        feature_specs (dict): Dictionary containing feature specifications.
                              Each key is a feature name, and each value is a dict with 'function' and 'params'.
    Returns:
        dict: A dictionary containing all the extracted features.
    """
    import numpy as np
    from features_methods import features
    from features_methods import tf_idf_features

    all_features = {}

    # Iterate over each feature specification
    for feature_name, feature_info in feature_specs.items():
        # Extract function name and parameters
        function_name = feature_info.get('function')
        params = feature_info.get('params', {})

        # Try to locate the feature function
        feature_func = globals().get(function_name)

        # If not found, look for the feature in the `features_methods` module
        if feature_func is None:
            feature_func = getattr(features, function_name, None)

        # If still not found, look for the feature in the `tf_idf_features` module
        if feature_func is None:
            feature_func = getattr(tf_idf_features, function_name, None)

        if feature_func:
            # Handle sentence-level features separately if needed
            if feature_name.startswith("sentence"):
                # Call the function with specified parameters
                feature_vector = feature_func(**params)

                # Aggregate features
                for key in feature_vector.keys():
                    for value in feature_vector[key].keys():
                        value_to_add = feature_vector[key][value]
                        if not isinstance(value_to_add, list):
                            value_to_add = [value_to_add]
                        all_features.setdefault((key, value), []).extend(value_to_add)
            else:
                # Call the function with specified parameters
                feature_vector = feature_func(**params)

                # Aggregate features
                for key, value in feature_vector.items():
                    # Convert NumPy arrays to lists for compatibility
                    if isinstance(value, np.ndarray):
                        value = value.tolist()

                    if key in all_features:
                        if not isinstance(all_features[key], list):
                            all_features[key] = [all_features[key]]

                        if isinstance(value, list):
                            all_features[key].extend(value)
                        else:
                            all_features[key].append(value)
                    else:
                        all_features[key] = value
        else:
            print(f"Feature function {function_name} not found for feature {feature_name}.")
            continue

    return all_features

