import numpy as np

class EMA_model(object):
    """ Exponential moving average.
    """
    def __init__(self):
        self.alpha = 0.5

    def fit(self, data, verbose=False):
        """ The 'fitting' of an exponential moving average model is here defined as a grid search for 
            an optimal decay parameter alpha.
        """
        best_accuracy = 0
        best_alpha = 0
        for alpha in np.linspace(0.1,0.9,9):
            self.alpha = alpha
            accuracy = self.test(data)
            if verbose:
                print("Alpha: {:0.1f}\tAccuracy: {:4.2f}%".format(alpha, accuracy*100))
            if accuracy > best_accuracy:
                best_alpha = alpha
                best_accuracy = accuracy
        self.alpha = best_alpha
        if verbose:
            print("Best alpha: " + str(self.alpha))
    
    def predict(self, data):
        # Exponential Moving Average
        pred_data = data["mid"]

        running_mean = 0.0
        run_avg_predictions = [running_mean]

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