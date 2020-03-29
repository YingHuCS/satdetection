import torch

from .hook import Hook


class EmptyCacheHook(Hook):

    def __init__(self, before_epoch=False, after_epoch=False, after_iter=False, after_train_epoch=False):
        self._before_epoch = before_epoch
        self._after_epoch = after_epoch
        self._after_iter = after_iter
        self._after_train_epoch = after_train_epoch


    def after_iter(self, runner):
        if self._after_iter:
            torch.cuda.empty_cache()

    def before_epoch(self, runner):
        if self._before_epoch:
            torch.cuda.empty_cache()

    def after_epoch(self, runner):
        if self._after_epoch:
            torch.cuda.empty_cache() 

    def after_train_epoch(self, runner):
        if self._after_train_epoch:
            torch.cuda.empty_cache()


