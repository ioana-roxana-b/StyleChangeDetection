import warnings
from test_scripts import test_pan, test
import parse_args
warnings.filterwarnings("ignore")

PAN = False

if __name__ == '__main__':
    if PAN:
        var_config = vars(parse_args.parse_args_pan())

        train_dataset_path = (var_config.get("train_dataset_path"))
        test_dataset_path = (var_config.get("test_dataset_path"))

        train_truth_path = (var_config.get("train_truth_path"))
        test_truth_path = (var_config.get("test_truth_path"))

        generate_features = (var_config.get("generate_features"))
        features_path_train = (var_config.get("features_path_train"))
        features_path_test = (var_config.get("features_path_test"))
        classifier_config_path = (var_config.get("classifier_config_path"))
        classifier_config_key = (var_config.get("classifier_config_key"))
        language = (var_config.get("language"))
        wan_config = (var_config.get("wan_config"))

        test_pan.test_pan(train_dataset_path=train_dataset_path, 
                          test_dataset_path=test_dataset_path,  
                          train_truth_path=train_truth_path, 
                          test_truth_path=test_truth_path,
                          generate_features=generate_features,
                          features_path_train=features_path_train, 
                          features_path_test=features_path_test, 
                          classifier_config_path=classifier_config_path,
                          classifier_config_key=classifier_config_key,
                          language=language,
                          wan_config=wan_config)

    else:
        var_config = vars(parse_args.parse_args())

        problem = (var_config.get("problem"))
        text_name = (var_config.get("text_name"))
        input_text_path = (var_config.get("input_text_path"))
        generate_features = (var_config.get("generate_features"))
        features_path = (var_config.get("features_path"))
        classifier_config_path = (var_config.get("classifier_config_path"))
        classifier_config_key = (var_config.get("classifier_config_key"))
        label = (var_config.get("label"))
        language = (var_config.get("language"))
        wan_config = (var_config.get("wan_config"))

        test.test(problem=problem,
                  text_name = text_name,
                  input_text_path=input_text_path,
                  generate_features=generate_features,
                  features_path=features_path,
                  classifier_config_path=classifier_config_path,
                  classifier_config_key=classifier_config_key,
                  label=label,
                  language=language,
                  wan_config=wan_config)