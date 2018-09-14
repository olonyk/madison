from sklearn import tree
import numpy as np


class DST_model(object):
    def __init__(self, max_depth=4):
        self.dtc = tree.DecisionTreeClassifier(random_state=0,
                                               criterion="entropy",
                                               max_depth=max_depth)
    
    def get_x(self, data):
        """ Format the x matrix in the form [sqrt(feature_i * feature_j)] for every feature and 
            every time stamp.
        """
        n_features = len(data.keys())
        x_temp = np.zeros((np.shape(data["mid"])[0]-1, n_features))
        x = np.zeros((np.shape(data["mid"])[0]-1, 2*(n_features**2)))
        for i, features in enumerate(sorted(data.keys())):
            x_temp[:, i] = data[features][:-1]
        for i, x_row in enumerate(x_temp):
            x[i, :n_features**2] = np.sqrt(np.reshape(np.dot(np.reshape(x_row, (np.size(x_row), -1)),
                                                np.reshape(x_row, (-1, np.size(x_row)))),
                                                (1, -1)))
            if i > 0:
                x[i, n_features**2:] = x[i-1, :n_features**2]
        return x
    
    def get_y(self, data):
        """ Format the y vector of the form [sign(t1 - t2), sign(t2 - t3), ...]
        """
        return np.sign(data["mid"][1:] - data["mid"][:-1])
    
    def fit(self, data):
        x = self.get_x(data)
        y = self.get_y(data)
        self.dtc.fit(x, y)
    
    def predict(self, data):
        x = self.get_x(data)
        return self.dtc.predict(x)

    def test(self, data, verbose=False):
        y = self.get_y(data)
        y_pred = self.predict(data)
        accuracy = np.sum(y_pred == y)/np.size(y)
        if verbose:
            print("{}\tAccuracy: {:4.2f}%".format(self.__class__.__name__, accuracy*100))
        return accuracy