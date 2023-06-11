import numpy as np
from typing import Tuple

def viterbi(x: np.ndarray, message: np.ndarray, H_hat: np.ndarray) -> Tuple[list, int]:
    """Viterbi algorithm

    Args:
        x (np.ndarray): Cover data
        message (np.ndarray): message bits array
        H_hat (np.ndarray): Generator matrix sub H

    Returns:
        Tuple[list, int]:
            First item is coded bits array
            Second item is embedding cost
    """
    h = len(H_hat)
    w = len(H_hat[0])
    wght = [float('inf')] * (2**h)
    wght[0] = 0
    indx = 0
    indm = 0
    rho = [1] * len(x)

    submatrices_number = len(message)
    path = [{} for _ in range(len(x))]

    for _ in range(submatrices_number):
        for j in range(w):
            new_wght = wght.copy()
            for k in range(2 ** h):
                w0 = wght[k] + x[indx] * rho[indx]
                curCol = int(''.join(str(h_hat_item) for h_hat_item in reversed(H_hat[:,j])), 2)
                w1 = wght[k ^ curCol] + (1 - x[indx]) * rho[indx]
                path[indx][k] = 1 if w1 < w0 else 0
                new_wght[k] = min(w0, w1)
            indx += 1
            wght = new_wght.copy()
        for j in range(2 ** (h - 1)):
            wght[j] = wght[2 * j + int(message[indm])]
        wght[2 ** (h - 1):2 ** h] = [float('inf')] * (2**h - 2**(h-1))
        indm += 1
        if indm >= len(message):
            break

    # Backward part
    indx -= 1
    indm -= 1
    embedding_cost = min(wght)
    state = wght.index(embedding_cost)
    state = 2 * state + int(message[indm])
    indm -= 1
    y = [0] * len(path)
    for _ in range(submatrices_number):
        for j in range(w - 1, -1, -1):
            y[indx] = path[indx].get(state) or 0
            curCol = int(''.join(str(h_hat_item) for h_hat_item in reversed(H_hat[:,j])), 2)
            state = state ^ (y[indx] * curCol)
            indx -= 1
        state = 2 * state + int(message[indm])
        indm -= 1
        if indm < 0:
            break
    return y, embedding_cost
