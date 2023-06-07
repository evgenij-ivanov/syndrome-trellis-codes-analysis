import numpy as np

# Define the generator matrix
G = np.array([[1, 0, 1, 1],
              [1, 1, 0, 1]])

# Define the code rate
code_rate = 1/2

# Generate a message of length 8
message = np.array([[1, 0, 1, 0],
                    [1, 1, 0, 1],
                    [0, 1, 0, 1],
                    [0, 0, 1, 1],
                    [1, 1, 0, 0],
                    [0, 1, 1, 1],
                    [1, 0, 1, 1],
                    [0, 0, 0, 0]])

def create_code_matrix(G, message, code_rate):
    # Split the message into fixed-length blocks
    block_length = int(len(message[0]) * code_rate)
    blocks = np.split(message, len(message) / block_length)

    # Generate the code matrix
    code_vectors = []
    for block in blocks:
        code = np.mod(block @ G, 2)
        code_vectors.append(code)
    code_matrix = np.concatenate(code_vectors, axis=0)