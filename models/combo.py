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
        return np.sign(data["mid"][:-1] - data["mid"][1:])
    
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
        self.sub_train_data = {name:data[:samples] for name, data in train_data.items()}
        train_data = {name:data[samples:] for name, data in train_data.items()}
        # Fit the submodels with their training data
        for model in self.models.values():
            model.fit(self.sub_train_data)
        # Get the predictions of the submodules of the train data
        y = self.get_y(train_data)
        predictions = np.zeros((np.size(y), len(self.models.keys())))
        for i, model_name in enumerate(sorted(self.models.keys())):
            predictions[:, i] = self.models[model_name].predict(train_data)
        # Include the raw data features in the model, this is questionable, consider changing to self.dtc.fit(predictions), see predict
        self.dtc.fit(self.get_x(predictions, train_data), y)
        self.train_data = train_data
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

        print("     \tSub  \tCombo\tTest\t  Agent")
        print("Model\tTrain\tTrain\tData\t  Simulation")
        print("━━━━━┳"+"━"*26+"┳"+"━"*10)
        for i, name in enumerate(sorted(self.models.keys())):
            # Sub train data
            st_acc = 100 * self.similarity(self.models[name].predict(self.sub_train_data), self.get_y(self.sub_train_data))
            ct_acc = 100 * self.similarity(self.models[name].predict(self.train_data), self.get_y(self.train_data))
            td_acc = 100 * np.sum(sub_pred[:, i] == y_true)/np.size(y_true)
            agent = self.agent(sub_pred[:, i], self.test_data["mid"])
            print("{}  ┃  {:4.2f}\t{:4.2f}\t{:4.2f}%  ┃  {:4.0f}".format(name, st_acc, ct_acc, td_acc, agent))
        print("━━━━━╋"+"━"*26+"╋"+"━"*10)
        
        st_sub_pred, st_combined_pred = self.predict(self.sub_train_data)
        st_y_true = self.get_y(self.sub_train_data)
        ct_sub_pred, td_combined_pred = self.predict(self.train_data)
        ct_y_true = self.get_y(self.train_data)
        agent_dst = self.agent(combined_pred, self.test_data["mid"])
        #print(self.get_opt_vec(sub_pred, y_true))
        #agent_opt = self.agent(self.get_opt_vec(sub_pred, y_true), self.test_data["mid"])
        agent_opt = self.agent(y_true, self.test_data["mid"])
        

        print("OPT  ┃  {:4.2f}\t{:4.2f}\t{:4.2f}%  ┃  {:4.0f}".format(
                                    self.get_opt(st_sub_pred, st_y_true),
                                    self.get_opt(ct_sub_pred, ct_y_true),
                                    self.get_opt(sub_pred, y_true),
                                    agent_opt))
        print("DST  ┃  {:4.2f}\t{:4.2f}\t{:4.2f}%  ┃  {:4.0f}".format(
                                    self.similarity(st_combined_pred, st_y_true)*100,
                                    self.similarity(td_combined_pred, ct_y_true)*100,
                                    self.similarity(combined_pred, y_true)*100,
                                    agent_dst))

        print("\nAgent Simulation")

        purse = self.agent(combined_pred, self.test_data["mid"])

        print("Agent simulator: {}".format(purse))

        print("\nSimilarity matrix:")
        names = sorted(self.models.keys())
        print("\t{}".format("\t".join(names)))
        for i, name in enumerate(names):
            sim = []
            for j in range(len(names)):
                sim.append(self.similarity(sub_pred[:, i], sub_pred[:, j]))
            print("{}\t{}".format(name, "\t".join("{:4.2f}".format(x) for x in sim)))

    def get_opt(self, sub_pred, y_true):
        opt = 0
        for y_t, y_p in zip(y_true, sub_pred):
            if y_t in y_p:
                opt += 1
        opt = opt/np.size(y_true)
        return opt*100
    
    def get_opt_vec(self, sub_pred, y_true):
        opt_vec = []
        for y_t, y_p in zip(y_true, sub_pred):
            if y_t in y_p:
                opt_vec.append(y_t)
            else:
                opt_vec.append(y_p[0])
        return opt_vec

    def similarity(self, x, y):
        return np.sum(x == y)/np.size(y)
    
    def agent(self, y_pred_vec, y_value_vec):
        purse = 0
        owned_stocks = 0
        bought = False
        for y_pred, y_value in zip(y_pred_vec, y_value_vec[1:]):
            if y_pred == 1 and not bought:
                owned_stocks = 1/y_value
                bought = True
            elif not y_pred == -1 and bought:
                purse += owned_stocks*y_value
                bought = False
        if bought:
            purse += owned_stocks*y_value_vec[-1]
        return purse
        