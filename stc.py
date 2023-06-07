import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ete3 import Tree
from ete3.coretype.tree import TreeNode
from typing import Union, List

cover_index: int = 0
message_index: int = 0
y: list = []
cover: Union[list, np.ndarray] = []
src_img_path: str = ''
encoded_img: Union[np.ndarray, None] = None
show_img: bool = True
tree: Union[Tree, None] = None
h: Union[np.ndarray, None] = None
sub_height: int = 0
sub_width: int = 0
message: str = ''


def reset_global_vars() -> None:
    global cover_index, message_index, y, tree, encoded_img, sub_height
    cover_index = 0
    message_index = 0
    y = []
    tree = init_trellis()
    encoded_img = []


def get_user_input() -> str:
    output = "Would you like to\n"
    output += "    (1) choose a message to hide\n"
    output += "    (2) generate random messages and see graphical representations of their embedding efficiencies\n"
    output += "    (3) generate random sub-matrix of different size and see graphical representations of " \
              + "their distortions\n"
    output += "    (4) generate random sub-matrix of same size and see graphical representations " \
              + "of their efficiencies\n"
    output += "    (5) look for the optimal choice of sub-matrix\n"
    while True:
        message_option = input(output)
        if not (message_option == '1' or message_option == '2' or message_option == '3' or message_option == '4'
                or message_option == '5'):
            print("Unrecognized input. Try again.")
        else:
            return message_option


def arbitrary_payload() -> None:
    global cover, h, sub_height, sub_width, message
    cover = select_img()
    sub_h = get_sub_h()
    print("Sub-matrix currently fixed at\n", np.asarray(sub_h))
    sub_height = len(sub_h)
    sub_width = len(sub_h[0])

    message = get_user_message()
    print("Generating matrix H...\n")
    h = get_h(sub_h, len(message), len(cover))
    print("H = \n" + str(h))

    reset_global_vars()
    embed()
    extract()

    distortion = calculate_distortion(Image.open(src_img_path).convert('L'), encoded_img)
    print("Distortion:", distortion)
    print("Message length:", len(message))
    print("Result in efficiency:", get_efficiency(len(message), distortion))


def random_payload_efficiencies() -> None:
    global cover, h, sub_height, sub_width, message, show_img
    show_img = False
    cover = select_img(13)

    sub_h = get_sub_h()
    print("Sub-matrix currently fixed at\n", np.asarray(sub_h))
    sub_height = len(sub_h)
    sub_width = len(sub_h[0])

    message_number = 30
    abscissa = []
    ordinate = []
    for inverse_alpha in range(10, 20 + 2, 2):
        print("1 / alpha = ", inverse_alpha)
        alpha = 1 / inverse_alpha
        message_length = math.floor(len(cover) * alpha)
        messages = get_random_payloads(message_number, message_length)
        h = get_h(sub_h, len(messages[0]), len(cover))
        efficiencies = []
        for i in range(message_number):
            print("    message", i, "/", message_number)
            message = messages[i]

            reset_global_vars()
            embed()

            distortion = calculate_distortion(Image.open(src_img_path).convert('L'), encoded_img)
            efficiencies.append(get_efficiency(message_length, distortion))
            print("Message length = ", message_length, ", distortion = ", distortion, ", efficiency = ",
                  efficiencies[i])
        abscissa.append(inverse_alpha)
        ordinate.append(np.median(np.asarray(efficiencies)))
    generate_graph("For n = " + str(len(cover)) + " sub_width = " + str(sub_width) + " sub_height = " + str(sub_height),
                   abscissa, ordinate, "1 / alpha", "efficiency")


def random_submatrix_distortions() -> None:
    global cover, h, sub_height, sub_width, message, show_img
    cover = select_img(13)
    show_img = False
    sizes = np.asarray([(2, 2), (3, 5), (4, 7), (6, 7)])
    sub_hs = []
    print("Generating sub-matrix")
    for s in sizes:
        sub_hs.append(get_random_sub_h(s[0], s[1]))
    abscissa = []
    ordinate = []
    alpha = 0.1
    message_length = math.floor(len(cover) * alpha)
    message = get_random_payloads(1, message_length)[0]
    for i in range(len(sub_hs)):
        print("Sub-matrix", i, "/", len(sub_hs))
        sub_h = sub_hs[i]
        sub_height = len(sub_h)
        sub_width = len(sub_h[0])
        print("Generating H")
        h = get_h(sub_h, len(message), len(cover))

        reset_global_vars()
        embed()

        distortion = calculate_distortion(Image.open(src_img_path).convert('L'), encoded_img)
        abscissa.append(i + 1)
        ordinate.append(distortion)

    x_label = "sizes: "
    for i in range(len(sizes)):
        x_label += "(" + str(sizes[i][0]) + "x" + str(sizes[i][1]) + ")"
        if i != len(sizes) - 1:
            x_label += ", "

    generate_graph("For n = " + str(len(cover)) + ", alpha = " + str(alpha), abscissa, ordinate, x_label, "distortion")


def random_submatrix_efficiencies() -> None:
    global cover, h, sub_height, sub_width, message, show_img
    cover = select_img(13)
    show_img = False
    sub_height = strict_integer_input("\nSub-matrix height: ")
    sub_width = strict_integer_input("Sub-matrix width: ")
    sub_hs = []
    submatrix_number = 100
    print("Generating sub-matrix")
    for i in range(submatrix_number):
        sub_hs.append(get_random_sub_h(sub_height, sub_width))
    abscissa = []
    ordinate = []
    alpha = 0.1
    message_length = math.floor(len(cover) * alpha)
    message = get_random_payloads(1, message_length)[0]
    for i in range(len(sub_hs)):
        print("sub-matrix", i, "/", len(sub_hs))
        sub_h = sub_hs[i]
        print("Generating H")
        h = get_h(sub_h, len(message), len(cover))

        reset_global_vars()
        embed()

        distortion = calculate_distortion(Image.open(src_img_path).convert('L'), encoded_img)
        efficiency = get_efficiency(message_length, distortion)
        abscissa.append(i + 1)
        ordinate.append(efficiency)
    ordinate = -np.sort(-np.asarray(ordinate))
    generate_graph("For n = " + str(len(cover)) + ", alpha = " + str(alpha), abscissa, ordinate,
                   "random sub-matrix sorted by efficiency", "efficiency")


def get_optimal_submatrix() -> None:
    global cover, h, sub_height, sub_width, message, show_img
    cover = select_img(13)
    show_img = False
    sub_height = strict_integer_input("\nSub-matrix height: ")
    sub_width = strict_integer_input("Sub-matrix width: ")
    sub_hs = []
    submatrix_number = 100
    print("Generating sub-matrix")
    for i in range(submatrix_number):
        sub_hs.append(get_random_sub_h(sub_height, sub_width))
    efficiencies = []
    alpha = 0.1
    message_length = math.floor(len(cover) * alpha)
    message = get_random_payloads(1, message_length)[0]
    for i in range(len(sub_hs)):
        print("sub-matrix", i, "/", len(sub_hs))
        sub_h = sub_hs[i]
        print("Generating H")
        h = get_h(sub_h, len(message), len(cover))

        reset_global_vars()
        embed()

        distortion = calculate_distortion(Image.open(src_img_path).convert('L'), encoded_img)
        efficiency = get_efficiency(message_length, distortion)
        efficiencies.append(efficiency)
    print("Best sub-matrix found:\n", sub_hs[np.argmax(efficiencies)])


def get_user_message() -> np.ndarray:
    def txt_to_bin(txt: str) -> np.ndarray:
        txt_bits = []
        char_to_code_mapping = {}
        for i in range(256):
            char_to_code_mapping[chr(i)] = i
        w = ""
        for c in txt:
            p = w + c
            if char_to_code_mapping.get(p) is not None:
                w = p
            else:
                char_to_code_mapping[p] = len(char_to_code_mapping)
                txt_bits.append(char_to_code_mapping[w])
                w = c
        txt_bits.append(char_to_code_mapping[w])
        message_bytes = np.empty(len(txt_bits) * 12, 'uint8')
        for i in range(len(txt_bits)):
            str_bits = format(txt_bits[i], '012b')
            for j in range(len(str_bits)):
                message_bytes[i * 12 + j] = str_bits[j]
        return message_bytes

    while True:
        txt_input = input("What would you like to hide today? ")
        bin_input = txt_to_bin(txt_input)
        if len(bin_input) > len(cover):
            print("\nThis message is too large for the selected cover! Try something shorter.")
        else:
            return bin_input


def get_random_payloads(message_number: int, message_length: int):
    return np.random.randint(0, 2, (message_number, message_length))


def strict_integer_input(output: str):
    while True:
        value = input(output + ' ')
        if not value.strip().isdigit():
            print("\nIntegers only, please.")
        else:
            break
    return int(value)


def strict_binary_input(output):
    while True:
        value = input(output + ' ').strip()
        non_binary = None
        for character in value:
            if not (character == '0' or character == '1'):
                non_binary = True
                break
        if non_binary or len(value) == 0:
            print("\nBase 2 numbers only, please.\n")
        else:
            break
    return int(value)


def select_img(cover_number=None):
    global src_img_path
    if cover_number is None:
        while True:
            cover_number = strict_integer_input("\nSelect image as cover [1-13]:")
            if cover_number > 13:
                print("\nUp to 13 only!")
            else:
                break
    src_img_path = 'img/' + str(cover_number) + '.jpg'
    img_bits = img_to_lsb()
    print("Cover: " + str(img_bits) + '\n')
    return img_bits


def img_to_lsb() -> np.ndarray:
    img: Union[Image, np.ndarray] = Image.open(src_img_path).convert('RGB')
    return np.mod(np.asarray(img), 2).flatten()


def get_random_sub_h(sub_h_height: int, sub_h_width: int) -> np.ndarray:
    sub_h = np.random.randint(0, 2, (sub_h_height, sub_h_width), "uint8")
    if not np.isin(1, sub_h[0]):
        sub_h[0][np.random.randint(sub_h_width)] = 1
    if not np.isin(1, sub_h[sub_h_height - 1]):
        sub_h[sub_h_height - 1][np.random.randint(sub_h_width)] = 1
    return sub_h


def get_efficiency(message_length: int, distortion: int):
    if distortion == 0:
        return message_length / 0.1
    return message_length / distortion


def get_sub_h() -> Union[np.ndarray, List[list]]:
    sub_h_height = strict_integer_input("\nSub-matrix height: ")
    sub_h_width = strict_integer_input("Sub-matrix width: ")
    print("Generating sub-matrix...\n")

    sub_h = get_random_sub_h(sub_h_height, sub_h_width)
    return sub_h


def get_h(sub_h, payload_size, message_size):
    global h
    sub_h_height = len(sub_h)
    sub_h_width = len(sub_h[0])
    h_width = message_size
    h_height = payload_size
    h = np.zeros((h_height, h_width), dtype='int8')

    def place_submatrix(h_row, h_column):
        cur_row: int
        for cur_row in range(sub_h_height):
            for cur_column in range(sub_h_width):
                if h_row + cur_row < h_height and h_column + cur_column < h_width:
                    h[h_row + cur_row][h_column + cur_column] = sub_h[cur_row][cur_column]

    for row in range(h_height):
        for column in range(h_width):
            if column == row * sub_h_width:
                place_submatrix(row, column)

    return h


def init_trellis():
    global tree, sub_height
    tree = Tree()
    root = tree.get_tree_root()
    root.add_features(y_bit='-')
    # noinspection PyTypeChecker
    first_node = tree.add_child(name='s' + ''.zfill(sub_height) + 'p0')
    first_node.add_features(dist=0, weight=0, state=''.zfill(sub_height), level='p0', y_bit='-')
    return tree


def get_column():
    global cover_index, h, sub_width, sub_height
    column = ''
    offset = int(cover_index / sub_width)
    for row in range(offset, offset + sub_height):
        if row == len(h):
            break
        column = column + str(h[row][cover_index])
    return column[::-1].zfill(sub_height)


def add_edge(cur_tree: Tree, node: TreeNode, y_bit: int):
    global cover_index
    cost = cover[cover_index] ^ y_bit
    weight = node.weight + cost

    next_state: str = ''

    if y_bit == 0:
        next_state = node.state
    elif y_bit == 1:
        column = get_column()
        next_state = str(bin(int(node.state, 2) ^ int(column, 2))[2:]).zfill(sub_height)

    existing_node = cur_tree.search_nodes(state=next_state, level=cover_index + 1)

    if len(existing_node):
        existing_node = existing_node[0]
        if existing_node.weight > weight:
            existing_node.detach()
            node.add_child(existing_node)
            existing_node.add_features(dist=cost, weight=weight, y_bit=y_bit)
    else:
        # noinspection PyTypeChecker
        new_node = node.add_child(name='s{0}c{1}'.format(str(next_state), str(cover_index + 1)), dist=cost)
        new_node.add_features(weight=weight, y_bit=y_bit, state=next_state, level=cover_index + 1)


def move_inside_block():
    global cover_index, sub_width, tree
    for i in range(sub_width):
        if i == 0:
            level = 'p' + str(message_index)
        else:
            level = cover_index
        column_nodes = tree.search_nodes(level=level)
        for node in column_nodes:
            add_edge(tree, node, 0)
            add_edge(tree, node, 1)
        cover_index += 1
    exit_block()


def exit_block():
    global cover_index, message_index, y, tree
    column_nodes = tree.search_nodes(level=cover_index)

    for node in column_nodes:
        if node.state[-1] == str(message[message_index]):
            connect_blocks(node)


def connect_blocks(node):
    global message_index, tree
    next_state = '0' + node.state[:-1]
    existing_node = tree.search_nodes(state=next_state, level=node.level + 1)

    if len(existing_node):
        existing_node = existing_node[0]
        if existing_node.weight > node.weight:
            existing_node.detach()
            node.add_child(existing_node)
            existing_node.add_features(weight=node.weight)
    else:
        new_node = node.add_child(dist=0, name='s' + next_state + 'p' + str(message_index + 1))
        new_node.add_features(state=next_state, level='p' + str(message_index + 1), weight=node.weight, y_bit='-')

        # check if last block
        if message_index == len(message) - 1:
            get_y(new_node)


def embed():
    global message, message_index
    for index in range(len(message)):
        message_index = index
        move_inside_block()


def get_y(node):
    global y, show_img
    while node:
        if isinstance(node.y_bit, int):
            y.insert(0, node.y_bit)
        node = node.up

    for i in range(len(y), len(cover)):
        y.append(cover[i])

    print("\nCalculating stego object done.")
    if show_img:
        print("Opening both images...")
    display_imgs()


def display_imgs():
    global encoded_img, src_img_path

    img: Union[np.ndarray, Image] = Image.open(src_img_path).convert('L')
    img_pixels = np.asarray(img, 'uint8')

    def get_stego_pixels():
        global src_img_path, cover
        encoded_pixels = []
        difference = np.absolute(y - cover)
        difference_matrix = vector_to_matrix(difference)

        for i in range(len(img_pixels)):
            encoded_pixels.append([])
            for j in range(len(img_pixels[0])):
                encoded_pixels[i].append(img_pixels[i][j] + difference_matrix[i][j])
                # Instead of adding 1, we decrement the maximum (255) by 1
                if encoded_pixels[i][j] == 256:
                    encoded_pixels[i][j] = 254
        return np.asarray(encoded_pixels, 'uint8')

    def vector_to_matrix(vector):
        matrix = []
        cover_rows = len(img_pixels)
        cover_columns = len(img_pixels[0])
        for i in range(cover_rows):
            matrix.append([])
            for j in range(cover_columns):
                matrix[i].append(vector[i * cover_rows + j])
        return matrix

    cover_img = Image.fromarray(img_pixels, 'L')
    if show_img:
        cover_img.show(title="Cover image")
    stego_pixels = get_stego_pixels()
    encoded_img = Image.fromarray(stego_pixels, 'L')
    if show_img:
        encoded_img.show(title="Stego image")


def extract():
    global h
    def bin_to_txt(bin_message: list) -> str:
        txt = ""
        code_char_mapping = {}
        for i in range(256):
            code_char_mapping[i] = chr(i)
        txt_bits = packed(bin_message)
        v = txt_bits[0]
        w = code_char_mapping[v]
        txt += w
        for i in range(1, len(txt_bits)):
            v = txt_bits[i]
            if code_char_mapping.get(v) is not None:
                entry = code_char_mapping[v]
            else:
                entry = w + w[0]
            txt += entry
            code_char_mapping[len(code_char_mapping)] = w + entry[0]
            w = entry
        return txt

    def packed(message_bytes):
        txt_bits = []
        for i in range(0, len(message_bytes), 12):
            txt_bits.append(int(''.join(np.array(message_bytes, '<U1')[i:i + 12]), 2))
        return txt_bits

    m = np.matmul(h, np.mod(np.asarray(encoded_img), 2).flatten())
    for i in range(len(m)):
        m[i] %= 2

    txt_output = bin_to_txt(list(m))

    print("\nMessage retrieved.")
    print("M = " + txt_output + '\n')


def generate_graph(title, x, y, x_label, y_label):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def calculate_distortion(cover_img, stego_img):
    return np.absolute(np.asarray(cover_img, np.int16).flatten() - np.asarray(stego_img, np.int16).flatten()).sum()


if __name__ == '__main__':

    print("\nHello! Welcome to our approach to PLS embedding using Syndrome-Trellis Coding.")
    print("We hope this command-line finds you well.\n")

    cover = select_img()
    print(cover.shape)
    sub_h = get_sub_h()
    print("Sub-matrix currently fixed at\n", np.asarray(sub_h))
    sub_height = len(sub_h)
    sub_width = len(sub_h[0])

    message = get_user_message()
    print("Generating matrix H...\n")
    h = get_h(sub_h, len(message), len(cover))
    print("H = \n" + str(h))

    reset_global_vars()
    embed()
    extract()

    distortion = calculate_distortion(Image.open(src_img_path).convert('L'), encoded_img)
    print("Distortion:", distortion)
    print("Message length:", len(message))
    print("Result in efficiency:", get_efficiency(len(message), distortion))
