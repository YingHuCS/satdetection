from collections import OrderedDict

import numpy as np


class TensorboardXBuffer(object):

    def __init__(self):
        self.output = OrderedDict()


    def update(self, vars):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            self.output[key] = var # ying

