import sklearn.svm as SVM
from sklearn.model_selection import train_test_split
import sklearn.metrics
import copy
from tqdm import tqdm
import warnings


warnings.filterwarnings('ignore',
                        category=sklearn.exceptions.UndefinedMetricWarning)


def to_selectvec(code, Xdim):
    bcode = format(code, '0%db' % Xdim)
    return list(map(int, list(bcode)))


def pick_andvalue(X, code):
    slist = list()
    for i, b in enumerate(code):
        if b == 1:
            slist.append(i)

    return X[:, slist]


def calc_score(targ, classifier, evaluator, Xdim,
               X_train, Y_train, X_test, Y_test):

    classifier_print = copy.deepcopy(classifier)
    # train classifier
    classifier_print.fit(pick_andvalue(
        X_train, to_selectvec(targ, Xdim)), Y_train)

    pred = classifier_print.predict(pick_andvalue(
        X_test, to_selectvec(targ, Xdim)))

    metric = evaluator(Y_test, pred)

    return [targ, ''.join(map(str, to_selectvec(targ, Xdim))), metric]


def calc_subset_wrapper(*params):
    queue, subset, classifier, evaluator, Xdim, X_train, Y_train, X_test, Y_test = params

    for targ in subset:
        queue.put(calc_score(targ, classifier, evaluator, Xdim,
                             X_train, Y_train, X_test, Y_test))


class Onesan(object):

    def __init__(self, X, Y, train_size=0.8, classifier=None,
                 classifier_param=None,
                 evaluator=lambda t, pred : sklearn.metrics.f1_score(t, pred, average='macro'),
                 n_onesan=1):

        # divide dataset into train and test
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(
                X, Y, test_size=1 - train_size
            )

        # number of dimensions
        self.Xdim = self.X_train.shape[-1]
        self.combinations = 2 ** self.Xdim

        # evaluation metrix
        self.evaluator = evaluator

        # number of parallel processes
        self.n_onesan = n_onesan

        # set classifier
        if classifier is not None:
            if not hasattr(classifier, 'fit') or \
                    not hasattr(classifier, 'predict'):
                raise TypeError(
                    'classifier must have "fit" and "predict" method.'
                )

            self.classifier = classifier

        else:
            if classifier_param is None:
                self.classifier = SVM.LinearSVC(C=0.05, random_state=1,
                                                max_iter=5000)
            else:
                self.classifier = SVM.LinearSVC(**classifier_param)

    def __run_single_onesan(self):
        result = list()

        for i in tqdm(range(1, self.combinations)):
            result.append(calc_score(
                i, self.classifier, self.evaluator, self.Xdim, self.X_train,
                self.Y_train, self.X_test, self.Y_test)
            )

        return result

    def __run_multiple_onesans(self):
        import multiprocessing as mp

        targets = range(1, self.combinations)
        cellsize = int(self.combinations / self.n_onesan)
        queue = mp.Queue()

        tasks = [
            (queue, targets[i * cellsize:i * cellsize + cellsize],
             self.classifier, self.evaluator, self.Xdim, self.X_train, self.Y_train,
             self.X_test, self.Y_test) for i in range(self.n_onesan)]

        ps = [mp.Process(target=calc_subset_wrapper, args=task)
              for task in tasks]

        for p in ps:
            p.start()

        result = list()
        for _ in tqdm(range(1, self.combinations)):
            result.append(queue.get())

        return result

    def run(self):

        if self.n_onesan <= 1:
            result = self.__run_single_onesan()

        else:
            result = self.__run_multiple_onesans()

        return sorted(result, key=lambda x: x[0])
