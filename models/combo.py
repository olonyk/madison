from sklearn import tree
import numpy as np

from .ema import EMA_model
from .svm import SVM_model
from .lep import LEP_model
from .dst import DST_model

class Combo(object):
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

        self.models = {"EMA": EMA_model(),
                       "SVM": SVM_model(),
                       "LEP": LEP_model(),
                       "DST": DST_model()}
        self.dtc = tree.DecisionTreeClassifier(random_state=0,
                                               criterion="entropy",
                                               max_depth=3)
    
    def get_y(self, data):
        """ Format the y vector of the form [sign(t1 - t2), sign(t2 - t3), ...]
        """
        return np.sign(data["mid"][1:] - data["mid"][:-1])
    
    def get_x(self, sub_pred, data):
        for name in sorted(data.keys()):
            sub_pred = np.concatenate((sub_pred, np.reshape(data[name], (np.size(data[name]), 1) )[:-1,:] ), axis=1)
        return sub_pred

    def fit(self, data=None):
        """ Fit the submodules with a subsection of the training data and train a 
            decision tree with the rest.
        """
        train_data = self.train_data if not data else data
        samples = int(np.shape(train_data["mid"])[0]*0.8)
        sub_train_data = {name:data[:samples] for name, data in train_data.items()}
        train_data = {name:data[samples:] for name, data in train_data.items()}
        # Fit the submodels with their training data
        for model in self.models.values():
            model.fit(sub_train_data)
        # Get the predictions of the submodules of the train data
        y = self.get_y(train_data)
        predictions = np.zeros((np.size(y), len(self.models.keys())))
        for i, model_name in enumerate(sorted(self.models.keys())):
            predictions[:, i] = self.models[model_name].predict(train_data)
        # Include the raw data features in the model, this is questionable, consider changing to self.dtc.fit(predictions), see predict
        self.dtc.fit(self.get_x(predictions, train_data), y)
        tree.export_graphviz(self.dtc,
                             out_file='combiner_tree.dot',
                             feature_names=list(sorted(self.models.keys())) + list(sorted(self.train_data.keys())),
                             class_names=["Decrease", "Unchanged", "Increase"],
                             filled=True,
                             rounded=True,  
                             special_characters=True)

    def predict(self, data=None):
        pred_data = self.test_data if not data else data
        sub_pred = np.zeros((np.size(self.get_y(pred_data)), len(self.models.keys())))
        for i, model_name in enumerate(sorted(self.models.keys())):
            sub_pred[:, i] = self.models[model_name].predict(pred_data)
        # Include the raw data features in the model, this is questionable, maybe
        # remove and change to self.dtc.predict(sub_pred). See fit.
        return sub_pred, self.dtc.predict(self.get_x(sub_pred, pred_data))
    
    def test(self, data=None):
        test_data = self.test_data if not data else data
        sub_pred, combined_pred = self.predict(test_data)
        y_true = self.get_y(test_data)

        print("-=Individual models=-")
        for i, name in enumerate(sorted(self.models.keys())):
            accuracy = np.sum(sub_pred[:, i] == y_true)/np.size(y_true)
            print("{}\t{:4.2f}%".format(name, accuracy*100))
        
        print("\n-=Combined models=-")        
        opt = 0
        for y_t, y_p in zip(y_true, sub_pred):
            if y_t in y_p:
                opt += 1
        opt = opt/np.size(y_true)
        print("Accuracy possible: {:4.2f}%".format(opt*100))
        print("Accuracy combined: {:4.2f}%".format(self.similarity(combined_pred, y_true)*100))


        print("\nSimilarity matrix:")
        names = sorted(self.models.keys())
        print("\t{}".format("\t".join(names)))
        for i, name in enumerate(names):
            sim = []
            for j in range(len(names)):
                sim.append(self.similarity(sub_pred[:, i], sub_pred[:, j]))
            print("{}\t{}".format(name, "\t".join("{:4.2f}".format(x) for x in sim)))

    def similarity(self, x, y):
        return np.sum(x == y)/np.size(y)