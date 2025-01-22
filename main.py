import warnings
from graph import pipeline_wan
from pan import pipeline_pan
from test_scripts import test_graph, test_pan

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # test.dialog(text = 'WinnieThePooh', config = 'kmeans', viz = 'tsne')
    # test.non_dialog(text='Crime_Anna', config='all', viz='tsne')

    pipeline_wan.pipeline_wan()
    test_graph.test_graph()

    # train_dir = "Corpus/pan14/pan14-authorship-verification-train-corpus-2014-04-22/pan14-authorship-verification-training-corpus-english-essays-2014-04-22"
    # test_dir = "Corpus/pan14/pan14-authorship-verification-test-corpus2-2014-04-22/pan14-authorship-verification-test-corpus2-english-essays-2014-04-22"
    # train_truth_file = "Corpus/pan14/pan14-authorship-verification-train-corpus-2014-04-22/pan14-authorship-verification-training-corpus-english-essays-2014-04-22/truth.txt"
    # test_truth_file = "Corpus/pan14/pan14-authorship-verification-test-corpus2-2014-04-22/pan14-authorship-verification-test-corpus2-english-essays-2014-04-22/truth.txt"
    # output_train_dir = "Corpus/pan/train-14-english-essays-corpus2"
    # output_test_dir = "Corpus/pan/test-14-english-essays-corpus2"
    #
    # pipeline_pan.pipeline_pan(train_dir, test_dir, train_truth_file, test_truth_file, output_train_dir, output_test_dir)
    #
    # test_pan.process_classifier()
