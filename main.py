import warnings

from test_scripts import test
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    test.dialog(text = 'WinnieThePooh', config = 'kmeans', viz = 'tsne')
    # test_tods.tods_test()