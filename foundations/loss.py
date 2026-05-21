import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        t1 = -1 / (len(y_true))
        L = 0
        for i in range(len(y_true)):
            y_pred[i] += 10**(-7)
            L +=((y_true[i]* np.log(y_pred[i])) + ((1-y_true[i])* np.log(1-y_pred[i])))
        L = t1 * L
        L = round(L, 4)
        return L


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # return round(your_answer, 4)
        t1 = -1 / (len(y_true))
        L = 0
        for i in range(len(y_true)): 
            for j in range(len(y_true[0])):
                y_pred[i][j] += 10**(-7)
                L += y_true[i][j] * np.log(y_pred[i][j])

        L = t1 * L 
        L = round(L,4)
        return L

        
