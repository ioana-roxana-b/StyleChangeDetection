import numpy as np

import features
import tf_idf_features


def save_features(text_data, feature_names, tf_idf_params=None):
    all_features = {}
    for feature_name in feature_names:
        feature_func = globals().get(feature_name)
        if feature_func is None:
            feature_func = getattr(features, feature_name, None)
        if feature_func is None:
            feature_func = getattr(tf_idf_features, feature_name, None)

        if feature_func:
            if feature_name.startswith("sentence"):
                feature_vector = feature_func(text_data)
                for key in feature_vector.keys():
                    for value in feature_vector[key].keys():
                        # Ensure the value is a list before extending.
                        value_to_add = feature_vector[key][value]
                        if not isinstance(value_to_add, list):
                            value_to_add = [value_to_add]  # Wrap non-lists in a list
                        all_features.setdefault(key, []).extend(value_to_add)
            else:
                if feature_name.startswith("tf_idf") and tf_idf_params is not None:
                    # Apply tf_idf_params if they are not None and the function supports it
                    feature_vector = feature_func(text_data, **tf_idf_params)
                else:
                    feature_vector = feature_func(text_data)

                # Convert numpy arrays to lists before storing in all_features
                for key, value in feature_vector.items():
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    all_features[key] = value
        else:
            print(f"Feature function {feature_name} not found.")
            continue

    return all_features


def save_sentence_features(text_data, feature_names):
    all_features = {}
    for feature_name in feature_names:
        feature_func = globals().get(feature_name)
        if feature_func is None:
            feature_func = getattr(features, feature_name, None)
        if feature_func:
            feature_vector = feature_func(text_data)
            for key in feature_vector.keys():
                for value in feature_vector[key].keys():
                    # Ensure the value is a list before extending.
                    value_to_add = feature_vector[key][value]
                    if not isinstance(value_to_add, list):
                        value_to_add = [value_to_add]  # Wrap non-lists in a list
                    all_features.setdefault(key, []).extend(value_to_add)
        else:
            print(f"Feature function {feature_name} not found.")
    return all_features

