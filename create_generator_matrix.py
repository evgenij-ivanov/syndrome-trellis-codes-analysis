import numpy as np

def create_generator_matrix(poly, constraint_length):
    # Convert polynomial coefficients to binary vector
    poly = np.array([int(x) for x in list(poly)], dtype=np.int8)

    # Compute number of output bits 
    n_outputs = 2**(constraint_length-1)

    # Initialize matrix with zeros
    G = np.zeros((n_outputs, constraint_length))

    # Set the first row of the matrix
    G[0, :] = np.concatenate((np.array([1]), np.zeros(constraint_length-2, dtype=np.int8), np.array([1])))

    # Compute the remaining rows of the matrix
    for i in range(1, n_outputs):
        feedback = np.mod(np.matmul(G[i-1,:], poly), 2)
        G[i,:] = np.concatenate((np.array([feedback]), G[i-1,:-1]))

    return G