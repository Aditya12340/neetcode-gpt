import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        n = max(z)
        sz = len(z)
        for k in range(sz):
            z[k] = z[k] - n
        e1 = np.exp(z)
        sum1 = 0
        final = []
        for i in range(sz):
            sum1 += e1[i]
        temp = 0
        for j in range(sz):
            temp = e1[j] / sum1
            temp = round(temp,4 )
            final.append(temp)
        
        return final

        


