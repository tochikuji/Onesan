import sys
from onesan import onesan
import pandas
import numpy as np
import pickle


data_df = pandas.read_csv('./winequality-white.csv')
data = data_df.as_matrix()

X = data[:, [0,1,2,3,4,5,6,7,8,9,10]]
Y = data[:, 11]

obj = onesan.Onesan(X, Y, n_onesan=4)
result = obj.run()

pickle.dump(result, open('wine.result.pkl', 'wb'))
