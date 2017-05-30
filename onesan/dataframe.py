# Dataset definition used by One-san

import pandas
import numpy
import os


class DataFrame(object):

    def __init__(self, data=None, **args):
        if data is None:
            data = args["data"]

        if isinstance(data, (numpy.array, tuple, list)):
            t = "array"
        elif isinstance(data, str):
            t = "file"
        else:
            t = None

        if type(args["type"]) is str:
            t = args["type"]

        if t is None:
            raise TypeError('invalid data type')
