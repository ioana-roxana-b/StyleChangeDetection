import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process CLI.')
    parser.add_argument('--problem',
                        type=str,
                        required=True,
                        help='There are three different pipelines available depending on the dataset: normal, dialogism, wan')
    parser.add_argument('--text-name',
                        type=str,
                        required=True,
                        help='The title of the text')
    parser.add_argument('--input-text-path',
                        type=str,
                        required=False,
                        help='Path to the text to be analyzed')
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
    parser.add_argument('--language',
                        type=str,
                        required=False,
                        default='en',
                        help='Required for wan problems')
    parser.add_argument('--wan-config',
                        type=str,
                        required=False,
                        default='C1',
                        help='Choose a config from the wan_configs file')

    args = parser.parse_args()
    return args

def parse_args_pan() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process CLI.')
    parser.add_argument('--train-dataset-path',
                        type=str,
                        required=False,
                        help='Path to the train dataset for PAN')
    parser.add_argument('--test-dataset-path',
                        type=str,
                        required=False,
                        help='Path to the test dataset for PAN')
    parser.add_argument('--train-truth-path',
                        type=str,
                        required=False,
                        help='Path to the train dataset ground truth for PAN')
    parser.add_argument('--test-truth-path',
                        type=str,
                        required=False,
                        help='Path to the test dataset ground truth for PAN')
    parser.add_argument('--generate-features',
                        type=bool,
                        required=False,
                        default=False,
                        help='Set True if there is no feature set generated for this problem')
    parser.add_argument('--features-path-train',
                        type=str,
                        required=True,
                        help='Path to the features of the train dataset')
    parser.add_argument('--features-path-test',
                        type=str,
                        required=True,
                        help='Path to the features of the test dataset')
    parser.add_argument('--classifier-config-path',
                        type=str,
                        required=True,
                        help='The name of the classification config to be used')
    parser.add_argument('--classifier-config-key',
                        type=str,
                        required=True,
                        help='What methods from the config should be used')
    parser.add_argument('--language',
                        type=str,
                        required=False,
                        default='en',
                        help='Required for wan problems')
    parser.add_argument('--wan-config',
                        type=str,
                        required=False,
                        default= 'C1',
                        help='Choose a config from the wan_configs file')

    args = parser.parse_args()
    return args
