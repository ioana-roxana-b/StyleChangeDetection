import features
import tf_idf_features

def save_chapter_features(text_data, feature_names, tf_idf_params=None):
    all_features = {}
    for feature_name in feature_names:
        feature_func = globals().get(feature_name)
        if feature_func is None:
            feature_func = getattr(features, feature_name, None)
        if feature_func is None:
            feature_func = getattr(tf_idf_features, feature_name, None)
        if feature_func and feature_name == "tf_idf_feature" and tf_idf_params is not None:
            feature_vector = feature_func(text_data, **tf_idf_params)
        elif feature_func:
            feature_vector = feature_func(text_data)
        else:
            print(f"Feature function {feature_name} not found.")
            continue

        for key, value in feature_vector.items():
            all_features.setdefault(key, []).extend(value if isinstance(value, list) else value)

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
                    all_features.setdefault(key, []).extend(value if isinstance(feature_vector[key][value], list) else feature_vector[key][value])
        else:
            print(f"Feature function {feature_name} not found.")
    return all_features
