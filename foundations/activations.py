import numpy as np
from numpy.typing import NDArray



class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        sigma = []
        for i in range(len(z)):
            x = 1 / (1 + (2.718281828459045**(-(z[i]))))
            x = round(x,5)
            sigma.append(x)
    
        return sigma

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        relu = []
        temp = 0
        for i in range(len(z)):
            temp = max(0,z[i])
            temp = float(temp)
            relu.append(temp)
            temp = 0
        return relu
