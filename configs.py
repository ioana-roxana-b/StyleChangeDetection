import features
import tf_idf_features

def config1():
    tf_idf_params = {
        "stop_words": False,
        "pos": False,
        "n_grams": True,
        "n": 2
    }
    feature_names = ["tf_idf_feature", "another_feature"]
    return feature_names, tf_idf_params