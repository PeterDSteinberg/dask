""" 
Tests for machine learning methods of:

    SGDClassifier
    SGDRegressor
    Perceptron
    PassiveAggressiveClassifier
    MultinomialNB
    BernoulliNB
    MiniBatchKMeans

TODO the following need to be tested:
    MiniBatchDictionaryLearning
    PassiveAggressiveRegressor
"""

from __future__ import division, print_function
from dask.ml.DaskML import *
from sklearn.datasets import load_boston
from sklearn.feature_extraction.image import extract_patches_2d
import dask.array as da
import numpy as np
import inspect
from pprint import pprint

def test_minibatchkmeans():

    rows,cols = 1000,5
    Xn = np.random.uniform(0, 
                    1,
                    rows * cols).reshape((rows, cols))
    Xn[::2,:] += 100.0
    Yn =  np.ones((rows, 1),dtype=np.int32)
    Yn[::2] = 0
    X = da.from_array(Xn, 
                    blockshape=(rows //10, 5))
    Y = da.from_array(Yn,
            shape=(Yn.size,),
            blockshape=(rows // 10, 1))
    kwargs_to_cls = {'max_iter': 20,
                    'n_clusters': 2,}
    args_to_cls = []
    global km
    km = MiniBatchKMeans(args_to_cls, 
                kwargs_to_cls, 
                X, 
                sample_gridx=100,
                batch_size=1000,
                n_batches=2,
                print_progress=False,
                Y=Y,
                shuffle=False)
    km.fit()
    print('Model: MiniBatchKMeans')
    print(km.stats['score_summary'])
    best_centroid = km.model.cluster_centers_[km.model.predict([100.0] * 5)[0]]
    mean_of_centroid = np.mean(best_centroid)
    assert mean_of_centroid > 100.0
    

ml_group1 = ('PassiveAggressiveClassifier',
            'SGDRegressor',
            'SGDClassifier',
            'Perceptron',
            )
def test_ml_1():
    global runner   
    rows = 10000
    cols = 5
    X = np.ones((rows, cols)) * 2
    X[::2,:] = 0
    X += np.random.uniform(0,.05, rows * cols).reshape((rows, cols))
    Y = np.ones((rows,))
    Y[::2] = 0
    batch_size = rows // 2
    x_block = batch_size // 10
    Y = da.from_array(Y,
                      shape=Y.shape, 
                      blockshape=(x_block,))
    X = da.from_array(X,
                     shape=X.shape, 
                     blockshape=(x_block, 1))
    for k in ml_group1:
        cls =  globals()[k]
        kwargs =    {
                    'sample_gridx': 200,
                    'batch_size': 4000,
                    'n_batches': 2,
                    'print_progress': False,
                    'Y': Y,
                    'shuffle': False,
                    }
        args_to_cls = []
        kwargs_to_cls = {}
        runner = cls(args_to_cls,
                    kwargs_to_cls,
                    X,
                    **kwargs)
        runner.get_all_classes = lambda: [0, 1]
        runner.fit()
        print('Model',k,'Score Summary:')
        pprint(runner.stats['score_summary'])
        assert runner.stats['score_summary']['mean'] > .92
            
default_kwargs = {
            'batch_size': 1000,
            'n_batches': 4,
            'print_progress': False,
            'sample_gridx': 500,
            'shuffle': False,
        }

def test_bernoullinb():
    global runner
    rows = 10000
    cols = 2
    X = np.zeros((rows, cols))
    X[1::2, 0] = 1
    X[0::2, 1] = 1
    X[1::2, 1] = 0
    X[0::2, 0] = 0
    Y = np.zeros((rows, 1))
    Y[::2] = 1
    X = da.from_array(X, 
                    shape=X.shape, 
                    blockshape=(100, 2))
    Y = da.from_array(Y, 
                    shape=Y.shape, 
                    blockshape=(100, 1))
    args_to_cls = []
    kwargs_to_cls = {}
    kwargs = default_kwargs.copy()
    kwargs['Y'] = Y
    runner = BernoulliNB(args_to_cls, 
                kwargs_to_cls, 
                X,
                **kwargs)
    runner.get_all_classes= lambda: [0, 1]
    runner.fit()
    print("BernoulliNB\n",runner)
    assert runner.stats['score_summary']['mean'] > .92


def test_multinomialnb():
    global runner
    rows = 100
    cols = 1000
    X = np.random.randint(rows, size=(rows, cols))
    Y = np.array(list(range(rows)))
    X[0,:] = rows - 1
    X = da.from_array(X, shape=X.shape, blockshape=(rows, cols))
    Y = da.from_array(Y, shape=Y.shape, blockshape=(rows, 1))
    args_to_cls = []
    kwargs_to_cls = {}
    kwargs = default_kwargs.copy()
    kwargs['Y'] = Y
    kwargs['sample_gridx'] = 5
    kwargs['batch_size'] = 50
    runner = MultinomialNB(args_to_cls, 
                kwargs_to_cls, 
                X,
                **kwargs)
    runner.get_all_classes= lambda: list(range(rows)) 
    runner.fit()
    print("MultinomialNB\n", runner)
    assert runner.stats['score_summary']['max'] > .95
    pred = runner.predict([rows - 1] * cols)[0]
    assert pred == 0
