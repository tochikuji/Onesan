import sklearn.svm as SVM
from sklearn.model_selection import train_test_split
import sklearn.metrics
import copy
from tqdm import tqdm


class Onesan(object):

    def __init__(self, X, Y, train_size=0.8, classifier=None, classifier_param=None):

        # divide dataset into train and test
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=1 - train_size
        )

        # number of dimensions
        self.Xdim = self.X_train.shape[-1]
        self.combinations = 2 ** self.Xdim

        # set classifier
        if classifier is not None:
            if not hasattr(classifier, 'fit') or not hasattr(classifier, 'predict'):
                raise TypeError('classifier must have "fit" and "predict" method.')

            self.classifier = classifier

        else:
            if classifier_param is None:
                self.classifier = SVM.LinearSVC(C=0.05, random_state=1, max_iter=5000)
            else:
                self.classifier = SVM.LinearSVC(**classifier_param)

    def to_selectvec(self, code):
        bcode = format(code, '0%db' % self.Xdim)
        return list(map(int, list(bcode)))

    def pick_andvalue(self, X, code):
        slist = list()
        for i, b in enumerate(code):
            if b == 1:
                slist.append(i)

        return X[:, slist]

    def run(self):

        result = list()

        for i in tqdm(range(1, self.combinations)):
            classifier = copy.deepcopy(self.classifier)
            # train classifier
            classifier.fit(self.pick_andvalue(self.X_train, self.to_selectvec(i)), self.Y_train)

            pred = classifier.predict(self.pick_andvalue(self.X_test, self.to_selectvec(i)))
            accuracy = sklearn.metrics.accuracy_score(pred, self.Y_test)

            result.append([i, ''.join(map(str, self.to_selectvec(i))), accuracy])

        return result
