import string
from random import choice
from sklearn.metrics import mean_squared_error
from common import message_to_binary, embed_to_img, binary_to_message
from PIL import Image
import argparse
import numpy as np
import os


embed_mode = 'embed'
extract_mode = 'extract'
evaluate_mode = 'evaluate'


def lsb_encode(image_path: str, message: str, stego_path: str):
    """Embed message into image using LSB method

    Args:
        image_path (str): Path to original image
        message (str): message to embed
        stego_path (str): path to stego image

    Raises:
        ValueError: Message cannot be embedded
    """
    img = Image.open(image_path)

    max_payload_size = np.array(img).flatten().shape[0]

    if message is None:
        message_size = max_payload_size // 8
        message = ''.join(choice(string.printable) for _ in range(message_size))

    # Convert message to binary
    binary_message = message_to_binary(message)

    # Check if message can fit into the image
    if len(binary_message) > max_payload_size:
        raise ValueError("Message is too large to be hidden in the image")

    # Encode the message in pixels
    data = img.getdata()
    embed_to_img(data, binary_message, stego_path, img.mode, img.size)
    

def lsb_decode(image_path: str) -> str:
    """Extract message from image

    Args:
        image_path (str): path to stego image

    Returns:
        str: embedded message
    """
    img = Image.open(image_path)
    data = img.getdata()

    binary_message = ''
    for pixel in data:
        r, g, b = '{0:08b}'.format(pixel[0]), '{0:08b}'.format(pixel[1]), '{0:08b}'.format(pixel[2])
        binary_message += r[-1] + g[-1] + b[-1]  # Extract LSB
    message = binary_to_message(binary_message)
    return message


def evaluate(cover_image_path: str, stego_image_path: str) -> str:
    """Evaluate mean squared error of message embedding

    Args:
        cover_image_path (str): path to original image
        stego_image_path (str): path to stego image

    Returns:
        str: mean squared error
    """
    cover_img = Image.open(cover_image_path)
    stego_image = Image.open(stego_image_path)
    cover_pixels = cover_img.getdata()
    stego_pixels = stego_image.getdata()
    mse = mean_squared_error(cover_pixels, stego_pixels)
    return mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='LSB', description='This app allows to embed a message into an image')

    parser.add_argument('--mode', help=f'"{embed_mode}", "{extract_mode}" or "{evaluate_mode}"', \
        choices=[embed_mode, extract_mode, evaluate_mode])
    parser.add_argument('--cover', help='Cover image path', type=argparse.FileType('r'))
    parser.add_argument('--message', help='Path to message text file', type=argparse.FileType('r'))
    parser.add_argument('--stegano', help='Path to stego object', type=argparse.FileType('w'))
    parser.add_argument('--embedded', help='Path to an image with embedded message', type=argparse.FileType('r'))
    
    args = parser.parse_args()

    mode, cover, message_file, stegano, embedded = args.mode, args.cover, args.message, args.stegano, args.embedded
    
    if mode == embed_mode:
        
        if not cover:
            cover = input('Type in cover image path: ')
            if not os.path.exists(cover):
                parser.error('cover image doesn\'t exist')
        else:
            cover = cover.name
        
        message = ''
        
        if not message_file:
            message = input('Type in message: ')
        else:
            message = message_file.read()
        
        if not stegano:
            stegano = input('Type in stegano image path: ')
        else:
            stegano = stegano.name    
        
        lsb_encode(cover, message, stegano)
    elif mode == extract_mode:
        
        if not embedded:
            embedded = input('Type in image with embedded message path: ')
            if not os.path.exists(embedded):
                parser.error('image with embedded message doesn\'t exist')
        else:
            embedded = embedded.name
        
        hidden_message = lsb_decode(embedded)
        print(f'Embedded message: {hidden_message}')
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
        print(f'MSE: {evaluation}')
