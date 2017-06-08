# やめて！おねえさん死んじゃう！！

<https://www.youtube.com/watch?v=Q4gTV4r0zRs>

# Onesan - The bruteforce feature selector

## REQUIREMENTS

- Python 3
- scikit-learn
- tqdm

# Instalaltion

`pip install .`

# License

MIT

# Usage

```
from onesan import onesan

# prepare the training and validation dataset
X = feature matrix # numpy.array
Y = target vector # numpy.array

# create onesan
robot = onesan(X, Y,
               train_size=0.9, # divide to x0.9 for training, x0.1 for validation
               n_onesan=8 # number of parallel processes, 1 by default
        ) # if classifier was not specified, onesan will use linear-SVM as a classifier by default

# Good luck! Onesan!!!!
result = onesan.run()

print(result)
'''
returns list of list
[
  [1, '0000...01', accuracy_1],
  [2, '0000...10', accuracy_2],
  ...,
  [2^d - 1, '1111...11', accuracy]
]
'''
```

# Reference
## Onesan

### initializer
`__init__(self, X, Y, train_size=0.8, n_onesan=1, classifier=None, classifier_param=None)`  

`n_onesan` specifies a number of onesans.  If `n_onesan == 1`, Onesan would run alone.
If `n_onesan >= 2`, Onesan would fission into child processes and runs almost `n_onesan` times faster.  
We can specify the classifier onesan uses.  
The `classifier` must have `fit` and `predict` method to training and validation
the model.  
`classifier` shold inherit `sklearn.base.BaseEstimator`.  

# Author

Aiga SUZUKI <ai-suzuki@aist.go.jp>
