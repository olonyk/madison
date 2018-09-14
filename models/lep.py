import numpy as np

class LEP_model(object):
    """ Linear Extrapolation.
        Very basic linear extrapolation with respect to the previous two data points.
    """
    def fit(self, data):
        """ No need for training Linear Extrapolation.
        """
        pass
    
    def predict(self, data):
        data = data["mid"]
        predictions = np.zeros((np.shape(data)[0]))
        predictions[2:] = ((data[1:-1] > data[:-2])*2)-1
        return predictions[1:]
