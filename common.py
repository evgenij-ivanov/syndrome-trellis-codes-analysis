import numpy as np
from PIL import Image
from typing import Tuple

def message_to_binary(message: str) -> str:
    """Get message as a binary string

    Args:
        message (str): message

    Returns:
        str: message as a binary string
    """
    binary_message = ''
    for char in message:
        binary_message += '{0:08b}'.format(ord(char))
    return binary_message


def binary_to_message(binary_message: str) -> str:
    """Convert message as a binary string to text message

    Args:
        binary_message (str): message as a binary string

    Returns:
        str: message as text
    """
    chars = []
    for i in range(0, len(binary_message), 8):
        char = chr(int(binary_message[i:i + 8], 2))
        if char != '\x00':
            chars.append(char)
    return ''.join(chars)


def embed_to_img(img_data: np.ndarray, emb_data: np.ndarray, out_image: str, img_mode: str, img_size: Tuple[int, int]):
    """Embed coded message bits to image and save resulting image

    Args:
        img_data (np.ndarray): Pixel values for original image
        emb_data (np.ndarray): Least bits values for embedding
        out_image (str): Result image path
        img_mode (str): Result image storing mode (RGB, RGBA, etc.)
        img_size (Tuple[int, int]): Size of result image as a tuple (width, height) in pixels
    """
    new_data = []

    j = 0
    for pixel in img_data:
        # Convert pixel color to binary
        r, g, b = '{0:08b}'.format(pixel[0]), '{0:08b}'.format(pixel[1]), '{0:08b}'.format(pixel[2])
        new_r, new_g, new_b = r[:-1], g[:-1], b[:-1]  # Remove last bit
        
        # Code embedding data into least bits
        if j < len(emb_data):
            new_r += str(emb_data[j])
        else:
            new_r = r
        j += 1
        
        if j < len(emb_data):
            new_g += str(emb_data[j])
        else:
            new_g = g
        j += 1
         
        if j < len(emb_data):
            new_b += str(emb_data[j])
        else:
            new_b = b
        j += 1
        
        # Append the new pixel to the list
        new_data.append((int(new_r, 2), int(new_g, 2), int(new_b, 2)))
    # Create a new image
    new_img = Image.new(img_mode, img_size)
    new_img.putdata(new_data)
    new_img.save(out_image)
