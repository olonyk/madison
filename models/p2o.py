import numpy as np

class P2O(object):
    """ Polynomial of second Order.
    """
    def fit(self, data):
        """ No need for training the curve fitting.
        """
        pass
    
    def predict(self, data):
        # Exponential Moving Average
        data = data["mid"]
        predictions = [running_mean]

        for pred_idx in range(1, pred_data.size):
            running_mean = running_mean*self.alpha + (1.0-self.alpha)*pred_data[pred_idx-1]
            run_avg_predictions.append(running_mean)
        
        # Create a vector with +1 and -1 where +1 indicates a predicted rise and vice verse.
        return [1 if x < y else -1 for x, y in zip(run_avg_predictions[:-1], run_avg_predictions[1:])]
    
    def test(self, data, verbose=False):
        y = self.get_y(data)
        y_pred = self.predict(data)
        accuracy = np.sum(y_pred == y)/np.size(y)
        if verbose:
            print("{}\tAccuracy: {:4.2f}%".format(self.__class__.__name__, accuracy*100))
        return accuracy
    
    def get_y(self, data):
        """ Format the y vector of the form [sign(t1 - t2), sign(t2 - t3), ...]
        """
        return np.sign(data["mid"][1:] - data["mid"][:-1])