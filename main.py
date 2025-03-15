import warnings
from graph import pipeline_wan
from pan import pipeline_pan
from test_scripts import test_graph, test_pan
import argparse

warnings.filterwarnings("ignore")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process CLI.')
    parser.add_argument('--problem',
                        type=str,
                        required=True,
                        help='There are four different pipelines available depending on the dataset: normal, dialogism, wan, pan')
    parser.add_argument('--input-text-path',
                        type=str,
                        required=False,
                        help='Path to the text to be analyzed')
    parser.add_argument('--train-dataset-path',
                        type=str,
                        required=False,
                        help='Path to the train dataset for PAN')
    parser.add_argument('--test-dataset-path',
                        type=str,
                        required=False,
                        help='Path to the test dataset for PAN')
    parser.add_argument('--generate-features',
                        type=bool,
                        required=False,
                        default=False,
                        help='Set True if there is no feature set generated for this problem')
    parser.add_argument('--features-path',
                        type=str,
                        required=True,
                        help='Path to the features of the text to be analysed')
    parser.add_argument('--classifier-config-path',
                        type=str,
                        required=True,
                        help='The name of the classification config to be used')
    parser.add_argument('--classifier-config-key',
                        type=str,
                        required=True,
                        help='What methods from the config should be used')
    parser.add_argument('--label',
                        type=str,
                        required=False,
                        default='Shakespeare|Fletcher|DOS|TOL',
                        help='Required for normal-text and wan problems')
    parser.add_argument('--visualisation',
                        type=bool,
                        required=False,
                        default=False,
                        help='Set true if you want to visualise the output')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    var_config = vars(parse_args())

    problem = (var_config.get("problem"))
    input_text_path = (var_config.get("input_text_path"))

    if problem == 'pan':
        train_dataset_path = (var_config.get("train_dataset_path"))
        test_dataset_path = (var_config.get("test_dataset_path"))

    generate_features = (var_config.get("generate_features"))
    features_path = (var_config.get("features_path"))
    classifier_config_path = (var_config.get("classifier_config_path"))
    classifier_config_key = (var_config.get("classifier_config_key"))
    label = (var_config.get("label"))
    visualisation = (var_config.get("visualisation"))

    print(var_config)

    # test.dialog(text = 'WinnieThePooh', config = 'kmeans', viz = 'tsne')
    # test.non_dialog(text='Crime_Anna', config='all', viz='tsne')

    # pipeline_wan.pipeline_wan()
    # test_graph.test_graph()
    #
    # train_dir = "Corpus/pan14/pan14-authorship-verification-train-corpus-2014-04-22/pan14-authorship-verification-training-corpus-spanish-articles-2014-04-22"
    # test_dir = "Corpus/pan14/pan14-authorship-verification-test-corpus2-2014-04-22/pan14-authorship-verification-test-corpus2-spanish-articles-2014-04-22"
    # train_truth_file = "Corpus/pan14/pan14-authorship-verification-train-corpus-2014-04-22/pan14-authorship-verification-training-corpus-spanish-articles-2014-04-22/truth.txt"
    # test_truth_file = "Corpus/pan14/pan14-authorship-verification-test-corpus2-2014-04-22/pan14-authorship-verification-test-corpus2-spanish-articles-2014-04-22/truth.txt"
    # output_train_dir = "Corpus/pan/train-14-spanish-articles-corpus2"
    # output_test_dir = "Corpus/pan/test-14-spanish-articles-corpus2"
    #
    # pipeline_pan.pipeline_pan(train_dir, test_dir, train_truth_file, test_truth_file, output_train_dir, output_test_dir)
    #
    # test_pan.process_classifier()