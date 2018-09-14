import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score

class SVM_model(object):
    def __init__(self):
        self.clf = svm.SVC()

    def get_x(self, data):
        """ Format the x matrix in the form [mid_t, mid_t-1, volume_t, volume_t-1]
        """
        x = np.zeros((np.shape(data["mid"])[0]-1, 4))
        x[:, 0] = data["mid"][:-1]
        x[0, 1] = x[0, 0]
        x[1:, 1] = data["mid"][:-2]

        x[:, 2] = data["volume"][:-1]
        x[0, 3] = x[0, 2]
        x[1:, 3] = data["volume"][:-2]
        return x
    
    def get_y(self, data):
        """ Format the y vector of the form [sign(t1 - t2), sign(t2 - t3), ...]
        """
        return np.sign(data["mid"][1:] - data["mid"][:-1])
    
    def fit(self, data):
        x = self.get_x(data)
        y = self.get_y(data)
        self.clf.fit(x, y)
    
    def predict(self, data):
        x = self.get_x(data)
        return self.clf.predict(x)

    def test(self, data, verbose=False):
        y = self.get_y(data)
        y_pred = self.predict(data)
        accuracy = np.sum(y_pred == y)/np.size(y)
        if verbose:
            print("{}\tAccuracy: {:4.2f}%".format(self.__class__.__name__, accuracy*100))
        return accuracy
