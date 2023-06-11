import viterbi
import numpy as np

y, cost = viterbi.viterbi(np.array([0,0,0,0,0,0,0,0]), np.array([0,0,1,1]), np.array([[1, 0], [1, 1]]))

print(y)

h = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
    ]

print(np.array(h).dot(y))
print(np.array(h).dot(np.array([0, 0, 1, 1, 1, 0, 0, 1])))