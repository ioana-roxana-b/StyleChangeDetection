import warnings

from graph import pipeline_wan
from test_scripts import test

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    #test.dialog(text = 'WinnieThePooh', config = 'kmeans', viz = 'tsne')
    #test.non_dialog(text='Crime_Anna', config='all', viz='tsne')

    pipeline_wan.pipeline_wan()