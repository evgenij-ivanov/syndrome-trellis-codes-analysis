
import cv2
import numpy as np
from pyldpc import make_ldpc, encode, decode

# Load an image
image = cv2.imread("img/cat.jpg")

# Get dimensions of image
height, width, channels = image.shape

# Set the number of pixels to embed
num_pixels = 96

# Calculate the number of information bits and the rate
k = num_pixels
rate = 0.25

# Calculate the number of coded bits and the codeword length
n = int(k / rate)
N = n - k

print(f'n = {n}')
print(f'k = {k}')
print(f'N = {N}')

# Generate a generator matrix for the LDPC code
G, H = make_ldpc(n, d_v=k, d_c=N, systematic=True, sparse=False)

# Convert the message to a numpy array
message = np.random.randint(2, size=(k,))

# Encode the message using the generator matrix
codeword = encode(message, G)

# Create a flat list of pixel values
pixels = image.flatten()

# Embed the codeword in the pixel values
idx = 0
for i in range(len(pixels)):
    # Check if we've embedded enough bits
    if idx >= len(codeword):
        break
    
    # Check if the pixel value is not 0 or 255
    if pixels[i] != 0 and pixels[i] != 255:
        # Increase or decrease the pixel value based on the codeword bit
        pixels[i] += 2 * int(codeword[idx]) - 1
        idx += 1

# Reshape the updated pixels back into an image
image_embedded = np.reshape(pixels, (height, width, channels))

# Save the modified image
cv2.imwrite("embedded_image.jpg", image_embedded)