import numpy as np

import features
import tf_idf_features

def save_features(text_data, feature_specs=None):
    all_features = {}
    for feature_name, params in feature_specs.items():
        feature_func = globals().get(feature_name)
        if feature_func is None:
            feature_func = getattr(features, feature_name, None)
        if feature_func is None:
            feature_func = getattr(tf_idf_features, feature_name, None)

        if feature_func:
            if feature_name.startswith("sentence"):
                feature_vector = feature_func(**params)
                for key in feature_vector.keys():
                    for value in feature_vector[key].keys():
                        value_to_add = feature_vector[key][value]
                        if not isinstance(value_to_add, list):
                            value_to_add = [value_to_add]
                        # Append values instead of overwriting
                        all_features.setdefault(key, []).extend(value_to_add)
            else:
                feature_vector = feature_func(**params if params else {})

                for key, value in feature_vector.items():
                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    # Append values instead of overwriting for keys that already exist
                    if key in all_features:
                        # Ensure existing data is in list form
                        if not isinstance(all_features[key], list):
                            all_features[key] = [all_features[key]]
                        # Append new data
                        if isinstance(value, list):
                            all_features[key].extend(value)
                        else:
                            all_features[key].append(value)
                    else:
                        all_features[key] = value
        else:
            print(f"Feature function {feature_name} not found.")
            continue

    return all_features


