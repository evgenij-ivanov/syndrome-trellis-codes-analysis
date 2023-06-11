import numpy as np
from PIL import Image
import argparse
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from viterbi import viterbi
from common import message_to_binary, binary_to_message, embed_to_img
import os


embed_mode = 'embed'
extract_mode = 'extract'
evaluate_mode = 'evaluate'


def get_random_sub_h(sub_height: int, sub_width: int) -> np.ndarray:
    """Generates random sub H matrix

    Args:
        sub_height (int): Height of generator matrix
        sub_width (int): Width of generator matrix

    Returns:
        np.ndarray: Generator matrix sub H
    """
    sub_h = np.random.randint(0, 2, (sub_height, sub_width), "uint8")
    if (not np.isin(1, sub_h[0])):
        sub_h[0][np.random.randint(sub_width)] = 1
    if (not np.isin(1, sub_h[sub_height - 1])):
        sub_h[sub_height - 1][np.random.randint(sub_width)] = 1
    return sub_h


def get_h(sub_h: np.ndarray, payload_size: int, cover_size: int) -> np.ndarray:
    """Creates parity matrix H

    Args:
        sub_h (np.ndarray): Generator matrix sub H
        payload_size (int): Size of payload (message)
        cover_size (int): Size of cover (image pixels amount)

    Returns:
        np.ndarray: Parity matrix H
    """
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

    return h.astype('uint8')


def evaluate(cover_image_path: str, stego_image_path: str) -> str:
    """Evaluate mean absolute error of message embedding

    Args:
        cover_image_path (str): path to original image
        stego_image_path (str): path to stego image

    Returns:
        str: mean absolute error
    """
    cover_img = Image.open(cover_image_path)
    stego_image = Image.open(stego_image_path)
    cover_pixels = np.array(cover_img.getdata()).flatten()
    stego_pixels = np.array(stego_image.getdata()).flatten()
    mae = mean_absolute_error(cover_pixels, stego_pixels)
    return mae


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='STC', description='This app allows to embed a message into an image')

    parser.add_argument('--mode', help=f'"{embed_mode}", "{extract_mode}" or "{evaluate_mode}"', \
        choices=[embed_mode, extract_mode, evaluate_mode], required=True)
    parser.add_argument('--cover', help='Cover image path', type=argparse.FileType('r'))
    parser.add_argument('--message', help='Path to message text file', type=argparse.FileType('r'))
    parser.add_argument('--stegano', help='Path to stego object', type=argparse.FileType('w'))
    parser.add_argument('--embedded', help='Path to an image with embedded message', type=argparse.FileType('r'))
    parser.add_argument('--h', help='Path to parity matrix file', type=argparse.FileType('r'))
    parser.add_argument('--h-save-file', help='Path to parity matrix file to save', type=argparse.FileType('w'))
    parser.add_argument('--sub-h-height', help='Sub H height', type=int)
    parser.add_argument('--sub-h-width', help='Sub H width', type=int)

    args = parser.parse_args()

    mode, cover, message_file, stegano, embedded, h_file, h_save_file, sub_h_height, sub_h_width = args.mode, args.cover, \
        args.message, args.stegano, args.embedded, args.h, args.h_save_file, args.sub_h_height, args.sub_h_width

    if mode == embed_mode:
        message = ''
        h = None

        if not cover:
            cover = input('Type in cover image path: ')
            if not os.path.exists(cover):
                parser.error('cover image doesn\'t exist')
        else:
            cover = cover.name

        img = Image.open(cover)
        data = np.array(img.getdata())
        x = np.mod(data.flatten(), 2)

        if not message_file:
            message = input('Type in message: ')
        else:
            message = message_file.read()

        k = len(x) // len(message)
        
        binary_message = message_to_binary(message)

        if not stegano:
            stegano = input('Type in stegano image path: ')

        if not h_file:
            is_custom_h = input('Would you like to type in your H hat matrix (y/n): ')
            if is_custom_h.lower() == 'y':
                sub_h_width = sub_h_width or k
                print(f'H hat width is {sub_h_width}')
                if not sub_h_height:
                    sub_h_height = int(input('Type in H hat matrix height: '))
                print('Type in H hat: ')
                sub_h = []
                for i in range(sub_h_height):
                    sub_h.append([int(e) for e in input().split()])
                sub_h = np.array(sub_h)
                h = sub_h # get_h(sub_h, len(binary_message), len(x))
            else:
                sub_h_height = sub_h_height or 8
                sub_h_width = sub_h_width or k
                sub_h = get_random_sub_h(sub_h_height, sub_h_width)
                h = sub_h # get_h(sub_h, len(binary_message), len(x))
        else:
            h = genfromtxt(h_file, delimiter=',')

        sub_h_width = sub_h_width or k
        print(f'H hat width is {sub_h_width}')

        y, cost = viterbi(x, binary_message, h[:sub_h_height, :sub_h_width])

        print(f'Embedding cost: {cost}')

        if h_save_file is not None:
            h = get_h(sub_h, len(binary_message), len(x))
            np.savetxt(h_save_file, h, delimiter=',', fmt='%i')

        result_y = np.copy(x)
        for i in range(len(y)):
            result_y[i] = y[i]
            
        embed_to_img(data, result_y, stegano, img.mode, img.size)
    elif mode == extract_mode:
        if not embedded:
            embedded = input('Type in stegano image path: ')
        else:
            embedded = embedded.name
         
        h = None
            
        if not h_file:
            h_file = input('Type in parity matrix H file path: ')
        
        h = genfromtxt(h_file, delimiter=',')
        
        if not sub_h_width:
            sub_h_width = int(input('Type in generator matrix sub H width: '))
        
        stegano_img = Image.open(embedded, 'r')
        stegano_data = np.array(stegano_img.getdata())
        y = np.mod(stegano_data.flatten(), 2)
        extracted_data = h.dot(y)
        encoded_binary_str = np.mod(extracted_data, 2).astype('uint8')
        encoded_y = ''.join(str(item) for item in encoded_binary_str)
        result = binary_to_message(encoded_y)
        print(result)
    elif mode == evaluate_mode:
        
        if not cover:
            cover = input('Type in cover image path: ')
            if not os.path.exists(cover):
                parser.error('cover image doesn\'t exist')
        else:
            cover = cover.name        
        
        if not embedded:
            embedded = input('Type in image with embedded message path: ')
            if not os.path.exists(embedded):
                parser.error('image with embedded message doesn\'t exist')
        else:
            embedded = embedded.name
        
        evaluation = evaluate(cover, embedded)
        print(f'MAE: {evaluation}')
