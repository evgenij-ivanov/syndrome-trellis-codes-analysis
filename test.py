import viterbi
import numpy as np
import common

for i in range(256):
    message = chr(i)
    binary_message = [int (c) for c in common.message_to_binary(message)]
    if (len(binary_message) < 0):
        continue
    y, cost = viterbi.viterbi(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), np.array(binary_message), np.array([[1, 0], [1, 1], [1, 0], [0, 1]]))

    print(y)

    h = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
        ]

    extracted_message = np.mod(np.array(h).dot(y), 2)
    print(extracted_message)
    print(binary_message)
    print('Yes' if np.array_equal(extracted_message, np.array(binary_message)) else 'No')

