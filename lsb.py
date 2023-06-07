import string
from random import choice
from sklearn.metrics import mean_squared_error

from PIL import Image


def message_to_binary(message):
    binary_message = ''
    for char in message:
        binary_message += '{0:08b}'.format(ord(char))
    return binary_message


def binary_to_message(binary_message):
    chars = []
    for i in range(0, len(binary_message), 8):
        char = chr(int(binary_message[i:i + 8], 2))
        if char != '\x00':
            chars.append(char)
    return ''.join(chars)


def lsb_encode(image_path, message):
    img = Image.open(image_path)

    if message is None:
        message_size = img.size[0] * img.size[1] * 3 // 8
        message = ''.join(choice(string.printable) for i in range(message_size))

    # Convert message to binary
    binary_message = message_to_binary(message)

    # Check if message can fit into the image
    if len(binary_message) > img.size[0] * img.size[1] * 3:
        raise ValueError("Message is too large to be hidden in the image")

    # Encode the message in pixels
    data = img.getdata()
    new_data = []
    message_index = 0
    for pixel in data:
        # Convert pixel color to binary
        r, g, b = '{0:08b}'.format(pixel[0]), '{0:08b}'.format(pixel[1]), '{0:08b}'.format(pixel[2])
        new_r, new_g, new_b = r[:-1], g[:-1], b[:-1]  # Remove last bit
        # Substitute LSB with message bits
        if message_index < len(binary_message):
            # print('R before:')
            # print(new_r)
            new_r += binary_message[message_index]
            # print('R after:')
            # print(new_r)
            message_index += 1
        else:
            new_r = r
        if message_index < len(binary_message):
            # print('G before:')
            # print(new_g)
            new_g += binary_message[message_index]
            # print('G after:')
            # print(new_g)
            message_index += 1
        else:
            new_g = g
        if message_index < len(binary_message):
            # print('B before:')
            # print(new_b)
            new_b += binary_message[message_index]
            # print('B after:')
            # print(new_b)
            message_index += 1
        else:
            new_b = b
        # Append the new pixel to the list
        new_data.append((int(new_r, 2), int(new_g, 2), int(new_b, 2)))
    # Create a new image
    new_img = Image.new(img.mode, img.size)
    new_img.putdata(new_data)
    new_img.save('encoded_image.png')  # Save the encoded image


def lsb_decode(image_path):
    img = Image.open(image_path)
    data = img.getdata()

    binary_message = ''
    for pixel in data:
        r, g, b = '{0:08b}'.format(pixel[0]), '{0:08b}'.format(pixel[1]), '{0:08b}'.format(pixel[2])
        binary_message += r[-1] + g[-1] + b[-1]  # Extract LSB
    message = binary_to_message(binary_message)
    return message


def evaluate(cover_image_path, stego_image_path):
    cover_img = Image.open(cover_image_path)
    stego_image = Image.open(stego_image_path)
    cover_pixels = cover_img.getdata()
    stego_pixels = stego_image.getdata()
    mse = mean_squared_error(cover_pixels, stego_pixels)
    print(f'MSE: {mse}')


original_image_path = 'img/1.jpg'
stego_image_path = 'encoded_image.png'
lsb_encode(original_image_path, 'asdsa')
hidden_message = lsb_decode(stego_image_path)
evaluate(original_image_path, stego_image_path)
print(hidden_message[:100])
