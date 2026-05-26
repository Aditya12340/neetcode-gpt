import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        x = np.dot(X,weights)
        y = np.round(x, 5)
        return y

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        t1 = 1 / len(model_prediction)
        x = model_prediction - ground_truth 
        x = x**2
        x = np.sum(x)
        x = t1 * x 
        x = round(x, 5)
        return x

        
