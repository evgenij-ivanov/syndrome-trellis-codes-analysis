import numpy as np
from PIL import Image
import argparse
from numpy import genfromtxt

def get_random_sub_h(sub_height, sub_width):
    sub_h = np.random.randint(0, 2, (sub_height, sub_width), "uint8")
    if (not np.isin(1, sub_h[0])):
        sub_h[0][np.random.randint(sub_width)] = 1
    if (not np.isin(1, sub_h[sub_height - 1])):
        sub_h[sub_height - 1][np.random.randint(sub_width)] = 1
    return sub_h


def get_h(sub_h, payload_size, cover_size):
    sub_height = len(sub_h)
    sub_width = len(sub_h[0])
    h_width = cover_size
    h_height = payload_size
    h = np.zeros((h_height, h_width), dtype='int8')

    def place_submatrix(h_row, h_column):
        for row in range(sub_height):
            for column in range(sub_width):
                if (h_row + row < h_height and h_column + column < h_width):
                    h[h_row + row][h_column + column] = sub_h[row][column]

    for row in range(h_height):
        for column in range(h_width):
            if (column == row * sub_width):
                place_submatrix(row, column)

    return h


def message_to_binary(message):
    binary_message = ''
    for char in message:
        binary_message += '{0:08b}'.format(ord(char))
    return binary_message


def viterbi(x, message, H_hat):
    h = len(H_hat)
    w = len(H_hat[0])
    wght = [float('inf')] * (2**h)
    wght[0] = 0
    indx = indm = 0
    rho = [1] * len(x)

    submatrices_number = min(len(x) // w, len(message))
    path = [dict() for _ in range(w * submatrices_number)]

    for i in range(submatrices_number):
        for j in range(w):
            new_wght = wght
            for k in range(2 ** h):
                w0 = wght[k] + x[indx] * rho[indx]
                curCol = int(''.join(str(h_hat_item) for h_hat_item in reversed(H_hat[:,j])), 2)
                w1 = wght[k ^ curCol] + (1 - x[indx]) * rho[indx]
                path[indx][k] = 1 if w1 < w0 else 0
                new_wght[k] = min(w0, w1)
            indx += 1
            wght = new_wght
        for j in range(2 ** (h - 1)):
            wght[j] = wght[2 * j + int(message[indm])]
        wght[2 ** (h - 1):2 ** h] = [float('inf')] * (2**h - 2**(h-1))
        indm += 1

    # Backward part
    embedding_cost = wght[0]
    state = 0
    indx -= 1
    indm -= 1
    y = [0] * len(path)
    for i in range(submatrices_number - 1, -1, -1):
        for j in range(w - 1, -1, -1):
            y[indx] = path[indx][state]
            curCol = int(''.join(str(h_hat_item) for h_hat_item in reversed(H_hat[:, j])), 2)
            state = state ^ (y[indx] * curCol)
            indx -= 1
        state = 2 * state + int(message[indm])
        indm -= 1
    return y, embedding_cost


def embed_to_img(img_data, emb_data, out_image):
    new_data = []

    j = 0
    for pixel in img_data:
        # Convert pixel color to binary
        r, g, b = '{0:08b}'.format(pixel[0]), '{0:08b}'.format(pixel[1]), '{0:08b}'.format(pixel[2])
        new_r, new_g, new_b = r[:-1], g[:-1], b[:-1]  # Remove last bit
        # Substitute LSB with message bits
        new_r += str(emb_data[j])
        new_g += str(emb_data[j + 1])
        new_b += str(emb_data[j + 2])
        # Append the new pixel to the list
        new_data.append((int(new_r, 2), int(new_g, 2), int(new_b, 2)))
        j += 3
    # Create a new image
    new_img = Image.new(img.mode, img.size)
    new_img.putdata(new_data)
    new_img.save(out_image)


parser = argparse.ArgumentParser(prog='STC', description='This app allows to embed a message into an image')

parser.add_argument('--cover', help='Cover image path')
parser.add_argument('--message', help='Path to message text file')
parser.add_argument('--stegano', help='Path to stego object')
parser.add_argument('--h', help='Path to parity matrix file')
parser.add_argument('--sub-h-height', help='Sub H height')

args = parser.parse_args()

cover, message_file, stegano, h_file, sub_h_height = args.cover, args.message, args.stegano, args.h, args.sub_h_height

message = ''
h = None

if not cover:
    cover = input('Type in cover image path: ')

img = Image.open(cover)
data = np.array(img.getdata())
x = np.mod(data.flatten(), 2)

if not message_file:
    message = input('Type in message: ')
else:
    with open(message_file, 'r') as f:
        message = f.read()

# k = len(x) // len(message)
k = 2
binary_message = message_to_binary(message)

if not stegano:
    stegano = input('Type in stegano image path: ')

if not h_file:
    is_custom_h = input('Would you like to type in your H hat matrix (y/n): ')
    if is_custom_h.lower() == 'y':
        print(f'H hat width is ${k}')
        if not sub_h_height:
            sub_h_height = int(input('Type in H hat matrix height: '))
        print('Type in H hat: ')
        sub_h = []
        for i in range(sub_h_height):
            sub_h.append([int(x) for x in input().split()])
        h = get_h(sub_h, len(binary_message), len(x))
    else:
        sub_h_height = 8
        sub_h = get_random_sub_h(sub_h_height, k)
        h = get_h(sub_h, len(binary_message), len(x))
else:
    h = genfromtxt(h_file, delimiter=',')

y, cost = viterbi(x, binary_message, h[:sub_h_height,:k])
result_y = x
for i in range(len(y)):
    result_y[i] = y[i]
extracted_data = h.dot(result_y)
encoded_binary_str = np.floor_divide(extracted_data, k)
encoded_y = ''.join(str(item) for item in encoded_binary_str)
result = ''
for i in range(0, len(encoded_y), 8):
    byte = encoded_y[i:i+8]
    result += chr(int(byte, 2))
print(result)

embed_to_img(data, encoded_x, stegano)

