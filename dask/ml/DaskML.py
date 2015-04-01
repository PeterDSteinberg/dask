from __future__ import print_function, division
import inspect
import itertools
import numpy as np
import random
import re
import sys
import tarfile
import time
from dask.array.core import rec_concatenate, Array
from functools import partial
from itertools import count
from operator import getitem
from pprint import pformat
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import get_data_home
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.externals.six.moves import html_parser
from sklearn.externals.six.moves import urllib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from toolz import merge

sample_names  = ('sample-%d' % i for i in count(1))
# all the algorithms to support partial_fit
partial_fit_algos = {
    'SGDClassifier': SGDClassifier,
    'SGDRegressor':SGDRegressor,
    'Perceptron': Perceptron,
    'PassiveAggressiveClassifier': PassiveAggressiveClassifier,
    'PassiveAggressiveRegressor':PassiveAggressiveRegressor,
    'MultinomialNB': MultinomialNB,
    'BernoulliNB': BernoulliNB,
    'MiniBatchDictionaryLearning': MiniBatchDictionaryLearning,
    'MiniBatchKMeans': MiniBatchKMeans,
}

class DaskMLBase(object): 
    """Classes by the same name as sklearn classes
    inherit from DaskMLBase.

    It standardizes the use of partial_fit methods 
    of sklearn classes for use with dask arrays. 

    Parameters:

        Positional args:
            args_to_cls: list/tuple
                args that go to the original sklearn class
                initialization, typically []
            kwargs_to_cls: dict 
                kwargs that go to the original sklearn class
                initialization. see sklearn docs
            X: 2-d dask array
                Training data X 
        Keywork args:
            Y: 1-d or 2-d dask array:
                True Y samples, 1- or 2-d,
                depending on sklearn class.
                Y.shape[0] === X.shape[0] 
            sample_weight: None/1-d dask array 
                sample weights to be applied in 
                training, if any.
                sample_weight.shape[0] == X.shape[0]
            sample_gridx: int 
                the regular blockshape (row count) 
                to which the X,Y, sample_weight
                should be reblocked.  Column blocking 
                is unaffected.
            batch_size: int
                How many rows to use for each sample 
                (This is automatically inserted in 
                    kwargs_to_cls when needed.)
            n_batches: int 
                How many batches of samples.
            print_progress: bool
                Print status on each batch
            callback: None/callable
                Run this on each batch
            shuffle: bool
                If True, shuffle among blocks using Poisson
                distribution (slow), else select blocks randomly
                using full slices of each block until sample
                size is met.
    """
    # These are the methods that require
    # all possible classes to be known in advance.
    requires_all_classes = ('MultinomialNB','BernoulliNB',
                            'Perceptron','SGDClassifier',
                            'PassiveAggressiveClassifier')
    cls_str = None # to over-ride this in each class
    def __init__(self, 
                args_to_cls, 
                kwargs_to_cls, 
                X, 
                Y=None,
                sample_weight=None,
                sample_gridx=500,
                batch_size=10000,
                n_batches=10,
                print_progress=False,
                callback=None,
                shuffle=False):
        # get the class object
        self.cls = partial_fit_algos[self.cls_str]
        # reblock the row dimension uniformly of X
        self.X = X.reblock(blockshape=(sample_gridx, X.shape[1])) 
        # reblock sample_weight if given
        if sample_weight is not None:
            self.sample_weight = sample_weight.reblock(blockshape=(sample_gridx,))
        else:
            self.sample_weight = None
        # determine whether Y (true samples)
        # is given and its dimension if so.
        if (Y is not None) and len(Y.shape) == 2:
            yb = (sample_gridx, Y.shape[1])
        else:
            yb = (sample_gridx,)
        # make sure X, Y, sample_weights
        # are all on same row blocks.
        if Y is not None:
            self.Y = Y.reblock(blockshape=yb)
        else:
            self.Y = None
        # save params
        self.sample_gridx = sample_gridx
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.print_progress = print_progress
        self.callback = callback
        # automate some initialization
        # of classes for ml
        try:
            self.spec = inspect.getargspec(
                            self.cls.__init__
                        ).args
        except:
            self.spec = {}
        if 'batch_size' in self.spec:
            kwargs_to_cls['batch_size'] = batch_size
        if 'shuffle' in self.spec:
            kwargs_to_cls['shuffle'] = False
        # initialize model
        self.model = self.cls(*args_to_cls, 
                                **kwargs_to_cls)
        # initialize scores and history
        try:
            self.score_spec = inspect.getargspec(
                                    self.model.score
                                ).args
        except:
            self.score_spec = {}
        self.stats = {
                     'iter_offset': 0, 
                     'score': 0.0,
                     'history': [], 
                     't0': time.time(),
                     'total_fit_time': 0.0,
            }
        

    def __repr__(self):
        s = pformat(self.stats)
        return "Model:\n%r\nStats:\n%s\n"%(self.model, s)
    def _row_getter(self, name, getarg, grab, twod=True):
        """Internal utility for forming slices 
        needed in dask array DAG's"""
        key = (name,0)
        if twod:
            key += (0,)
            slic_arg = lambda g: (slice(g,None,None),slice(None,None,None))
        else:
            slic_arg = lambda g: slice(g,None,None)
        if grab is None:
            slic = (slice(None, None, None),)*2
            return (key, (getitem, getarg, slic))
        kv = ( key, 
            (rec_concatenate,
                [(getitem, getarg, slic_arg(g)) for g in grab],
            )
        )
        return kv
    def get_sample(self, shuffle=False):
        """get_sample(shuffle=False)
        If shuffle is True,then the 
        Poisson distribution is used 
        to sample randomly within all blocks on 
        each sample (slow).  If shuffle is False, then
         blocks are selected at random using full 
         slices of each block until the sample 
         size is met.

         """
        cat_argx = []
        cat_argy = []
        cat_argsw = []
        # Poisson number
        M = self.batch_size * self.sample_gridx / self.X.shape[0]
        M = int(np.ceil(M))
        xname = next(sample_names)
        yname = next(sample_names)
        if self.sample_weight:
            swname = next(sample_names)
        row_count = 0
        choices = tuple(range(self.sample_gridx))
        # all possible row blocks
        blks = list(range(len(self.X.blockdims[0])))
        np.random.shuffle(blks)
        if shuffle:
            # choose small number from each block
            limit = len(blks)
        else:
            # choose all from small number of blocks
            limit = self.batch_size / self.sample_gridx
            limit = int(np.ceil(limit))
        for blk_row_idx in blks[:limit]:
            dskx, dsky, dsksw = {},{},{}
            indx =  (self.X.name, blk_row_idx, 0)
            if shuffle:
                # how many from this block
                grab_count = np.random.poisson(M)
                if not grab_count:
                    # chose 0 skip
                    continue
                row_count += grab_count
                # random list of row inds
                grab = [np.random.choice(choices) for _ in range(grab_count)]
            else:
                # do full slice of this block
                grab = None 
                grab_count = self.sample_gridx
            k,v = self._row_getter(xname, 
                                   indx,
                                   grab,
                                   twod=True)
            dskx[k] = v 
            if self.Y is not None:
                indy = (self.Y.name, ) + indx[1:len(self.Y.shape) + 1]
                k,v = self._row_getter(yname, 
                                      indy,
                                      grab,
                                      twod=len(self.Y.shape)==2)
                dsky[k] = v 
            if self.sample_weight:
                indsw = (self.sample_weight.name,) + (indx[1],)
                k,v = self._row_getter(swname, 
                                        indsw,
                                        grab,
                                        twod=False)
                dsksw[k] = v
                cat_argsw.append(Array(merge(dsksw, self.sample_weight.dask),
                    swname,
                    shape=(grab_count,1),
                    blockshape=(grab_count,1)
                    ).compute())
            cat_argx.append(Array(merge(dskx, self.X.dask),
                    xname,
                    shape=(grab_count, self.X.shape[1]),
                    blockshape=(grab_count, self.X.shape[1])).compute())
            if self.Y is not None:
                if len(self.Y.shape) == 2:
                    y_blockshape = (grab_count, self.Y.shape[1])
                else:
                    y_blockshape = (grab_count,)
                cat_argy.append(Array(merge(dsky, self.Y.dask),
                    yname,
                    shape=y_blockshape,
                    blockshape=y_blockshape).compute())

        sample_x = np.concatenate(cat_argx)
        if self.Y is not None:
            sample_y = np.concatenate(cat_argy)
        else:
            sample_y = None
        if self.sample_weight:
            sample_weight = np.concatenate(cat_argsw)
        else:
            sample_weight = None
        return (self.transform_x(sample_x), 
                    self.transform_y(sample_y), 
                    sample_weight)
    def transform_x(self, x):
        """Override if needed to transform 
        a 2-d numpy array X that is a sample
        from dask X."""
        return x 
    def transform_y(self, y):
        """Override if needed to transform
        a 1-d numpy array Y that is a sample from 
        dask Y."""
        return y
    def get_all_classes(self):
        """ Overriding this method is required 
        if using an sklearn method that requires 
        all possible classes to be known in advance."""
        raise ValueError(
                """Override get_all_classes to return 
all possible classes as list.  Required for %r
See sklearn docs. """ % self.cls)

    def partial_fit(self):
        """ 
        partial_fit(self)

        Standardizes arguments to partial_fit methods of 
        sklearn classes to allow X, Y, and sample_weight 
        to be dask arrays.

        """
        samp = self.get_sample(shuffle=self.shuffle)
        sample_x, sample_y, sample_weight = samp
        if self.cls_str in self.requires_all_classes:
            kwargs = {'classes': self.get_all_classes()}
        else:
            kwargs = {}
        if 'sample_weight' in self.spec and self.sample_weight is not None:
            kwargs['sample_weight'] = sample_weight
        if 'iter_offset' in self.spec:
            kwargs['iter_offset'] = self.stats['iter_offset']
        if not kwargs:
            self.model.partial_fit(sample_x, sample_y)
        else:
            self.model.partial_fit(sample_x, sample_y, **kwargs)
        self.stats['iter_offset'] += sample_x.shape[0]
        if hasattr(self.model, 'score'):
            score_args = [sample_x]
            if 'y' in self.score_spec or 'Y' in self.score_spec:
                score_args.append(sample_y)
            if 'sample_weight' in self.score_spec:
                self.stats['score'] = self.model.score(*score_args, 
                                        sample_weight=sample_weight)
            else:
                self.stats['score'] = self.model.score(*score_args)
            self.stats['history'].append((self.stats['iter_offset'], self.stats['score']))
        self.stats['total_fit_time'] = time.time() - self.stats['t0']
        if callable(self.callback): 
            self.callback()
    
    def fit(self):
        for step in range(self.n_batches):
            self.partial_fit()
            self.step = step
            if self.stop():
                break
        scores = ([h[1] for h in self.stats['history']])
        self.stats['score_summary'] = {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'min': np.min(scores),
            'max': np.max(scores),
        }
    def stop(self):
        """ Override this if needed.
        Return True to stop after a partial_fit
        step.
        """
        return False
    

class SGDClassifier(DaskMLBase):
    cls_str = 'SGDClassifier'


class SGDRegressor(DaskMLBase):
    cls_str = 'SGDRegressor'


class Perceptron(DaskMLBase):
    cls_str = 'Perceptron'


class PassiveAggressiveClassifier(DaskMLBase):
    cls_str = 'PassiveAggressiveClassifier'


class PassiveAggressiveRegressor(DaskMLBase):
    cls_str = 'PassiveAggressiveRegressor'


class MultinomialNB(DaskMLBase):
    cls_str = 'MultinomialNB'


class BernoulliNB(DaskMLBase):
    cls_str = 'BernoulliNB'


class MiniBatchDictionaryLearning(DaskMLBase):
    cls_str = 'MiniBatchDictionaryLearning'


class MiniBatchKMeans(DaskMLBase):
    cls_str = "MiniBatchKMeans"


a = partial_fit_algos.keys()
a.extend(('DaskMLBase','partial_fit_algos'))
__all__ = a
