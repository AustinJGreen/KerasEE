import gzip
import os
from random import randint
from shutil import copyfile, rmtree

import numpy as np
from PIL import Image
from skimage.draw import ellipse
from skimage.draw import line

from src import blocks


def load_minimap_values(base_dir):
    minimap_dict = {}
    with open('%s\\block_colors.txt' % base_dir) as fp:
        line = fp.readline()
        while line:
            space_index = line.index(' ')
            key_str = line[:space_index]
            argb_str = line[space_index:]
            key = int(key_str)
            argb = int(argb_str)
            minimap_dict[key] = argb
            line = fp.readline()
    return minimap_dict


def load_block_images(base_dir):
    block_dict = {}
    for filename in os.listdir('%s\\blocks' % base_dir):
        file = '%s\\blocks\\%s' % (base_dir, filename)
        if os.path.isfile(file):
            block_dict[int(filename[1:-4])] = Image.open(file)
    return block_dict


def load_encoding_dict(base_dir, name):
    block_forward_dict = {}
    block_backward_dict = {}
    with open('%s\\%s.txt' % (base_dir, name)) as fp:
        line = fp.readline()
        while line:
            block = int(line)
            assert block not in block_forward_dict
            block_hash = len(block_forward_dict)
            block_forward_dict[block] = block_hash
            block_backward_dict[block_hash] = block
            line = fp.readline()
    assert len(block_forward_dict) == len(block_backward_dict)
    return block_forward_dict, block_backward_dict


def load_label_dict(base_dir, name):
    label_dict = {}

    with open('%s\\%s.txt' % (base_dir, name)) as fp:
        line = fp.readline()
        while line:
            split_index = line.rindex(' ')
            world = line[:split_index]
            label = int(line[split_index + 1:])
            label_dict[world] = label
            line = fp.readline()

    return label_dict


def save_label_dict(base_dir, name, label_dict):
    with open('%s\\%s.txt' % (base_dir, name), 'w') as fp:
        for world_id in label_dict.keys():
            label = label_dict[world_id]
            fp.write('%s %s\n' % (world_id, label))


def convert_labels(raw_labels, categories=10, epsilon=1e-10):
    # Use label smoothing
    # http://www.deeplearningbook.org/contents/regularization.html
    # Section 7.5.1
    fill_value = epsilon / (categories - 1)
    hot_value = 1 - epsilon
    label_count = raw_labels.shape[0]

    soft_labels = np.full((label_count, categories), fill_value=fill_value, dtype=float)

    for label in range(label_count):
        soft_labels[label, raw_labels[label]] = hot_value

    return soft_labels


def convert_labels_binary(raw_labels, epsilon=1e-10):
    hot_value = 1 - epsilon
    label_count = raw_labels.shape[0]
    soft_labels = np.full((label_count, 1), fill_value=epsilon, dtype=float)

    for label in range(label_count):
        if raw_labels[label] == 1:
            soft_labels[label] = hot_value

    return soft_labels


def random_mask_high(width, height, channels=10):
    '''Generates a random irregular mask based off of rectangles'''
    mask = np.zeros((width, height, channels), np.int8)

    max_size = 12

    # Random rectangles
    for i in range(randint(1, 2)):
        x1 = randint(1, width - max_size - 1)
        x_width = randint(5, min(max_size, width - x1))
        x2 = x1 + x_width

        y1 = randint(1, height - max_size - 1)
        y_height = randint(5, min(max_size, height - y1))
        y2 = y1 + y_height

        mask[x1:x2, y1:y2, :] = 1

    # Random circles
    for j in range(randint(1, 2)):
        x1 = randint(1, width - max_size - 1)
        x_width = randint(5, min(max_size, width - x1))

        y1 = randint(1, height - max_size - 1)
        y_height = randint(5, min(max_size, height - y1))
        rr, cc = ellipse(y1, x1, y_height, x_width)
        mask[cc, rr, :] = 1

    # Random lines
    for j in range(randint(15, 20)):
        x1, x2 = randint(1, width - 1), randint(1, width - 1)
        y1, y2 = randint(1, height - 1), randint(1, height - 1)
        rr, cc = line(y1, x1, y2, x2)
        mask[cc, rr, :] = 1

    return 1 - mask


def random_mask_low(height, width, channels=11):
    '''Generates a random irregular mask based off of rectangles'''
    mask = np.zeros((height, width, channels), np.int8)

    max_size = 20

    # Random rectangles
    x1 = randint(1, width - max_size - 1)
    x_width = randint(12, min(max_size, width - x1))
    x2 = x1 + x_width

    y1 = randint(1, height - max_size - 1)
    y_height = randint(12, min(max_size, height - y1))
    y2 = y1 + y_height

    mask[x1:x2, y1:y2, :] = 1

    return mask


def mask_batch_high(batch):
    masked = np.empty(batch.shape, dtype=np.int8)
    masks = np.empty(batch.shape, dtype=np.int8)

    batch_size = batch.shape[0]
    for i in range(batch_size):
        world_masked = np.copy(batch[i])
        mask = random_mask_high(world_masked.shape[0], world_masked.shape[1], world_masked.shape[2])
        world_masked[mask == 0] = 1
        masked[i] = world_masked
        masks[i] = mask

    return masked, masks


def mask_batch_low(batch):
    masked = np.empty(batch.shape, dtype=np.int8)
    masks = np.empty(batch.shape, dtype=np.int8)

    batch_size = batch.shape[0]
    for i in range(batch_size):
        world_masked = np.copy(batch[i])
        mask = random_mask_low(world_masked.shape[0], world_masked.shape[1], world_masked.shape[2])
        world_masked[mask == 0] = 1
        masked[i] = world_masked
        masks[i] = mask

    return masked, masks


def decode_world_sigmoid(block_backward, world_data):
    bits = world_data.shape[2]
    width = world_data.shape[0]
    height = world_data.shape[1]
    world_copy = np.empty((width, height), dtype=int)
    for y in range(height):
        for x in range(width):
            value = 0
            for bit in range(bits):
                bit_data = world_data[x, y, bit]

                bit_value = 0
                if bit_data >= 0.5:
                    bit_value = 1
                value = value | (bit_value << ((bits - 1) - bit))

            if value in block_backward:
                value = block_backward[value]
            else:
                value = 0

            world_copy[x, y] = int(value)
    return world_copy


def decode_world_tanh(block_backward, world_data):
    bits = world_data.shape[2]
    width = world_data.shape[0]
    height = world_data.shape[1]
    world_copy = np.empty((width, height), dtype=int)
    for y in range(height):
        for x in range(width):
            value = 0
            for bit in range(bits):
                bit_data = world_data[x, y, bit]

                bit_value = 0
                if bit_data > 0:
                    bit_value = 1
                value = value | (bit_value << ((bits - 1) - bit))

            if value in block_backward:
                value = block_backward[value]
            else:
                value = 0

            world_copy[x, y] = int(value)
    return world_copy


def encode_world_sigmoid(block_forward, world_data):
    width = world_data.shape[0]
    height = world_data.shape[1]
    bits = 10
    world_copy = np.empty((width, height, bits), dtype=np.int8)

    for y in range(height):
        for x in range(width):
            value = int(world_data[x, y])
            if value in block_forward:
                value = block_forward[value]
            else:
                value = 0
            for bit in range(bits):
                bit_value = (value >> bit) & 1
                world_copy[x, y, bits - 1 - bit] = bit_value  # [0, 1]

    return world_copy


def encode_world_tanh(block_forward, world_data):
    width = world_data.shape[0]
    height = world_data.shape[1]
    bits = 10
    world_copy = np.empty((width, height, bits), dtype=np.int8)

    for y in range(height):
        for x in range(width):
            value = int(world_data[x, y])
            if value in block_forward:
                value = block_forward[value]
            else:
                value = 0
            for bit in range(bits):
                bit_value = (value >> bit) & 1
                bit_value_reshaped = (bit_value * 2) - 1
                world_copy[x, y, bits - 1 - bit] = bit_value_reshaped  # [-1, 1]

    return world_copy


def encode_world_minimap(minimap_values, world_data):
    width = world_data.shape[0]
    height = world_data.shape[1]

    encoded_values = np.empty((width, height, 3), dtype=float)
    for x in range(width):
        for y in range(height):
            block = int(world_data[x, y])
            if block in minimap_values:
                v = minimap_values[block]
                a = (v >> 24) & 0xFF
                r = (v >> 16) & 0xFF
                g = (v >> 8) & 0xFF
                b = v & 0xFF
                encoded_values[x, y, 0] = r / 255.0
                encoded_values[x, y, 1] = g / 255.0
                encoded_values[x, y, 2] = b / 255.0
            else:
                encoded_values[x, y, 0] = 0.0
                encoded_values[x, y, 1] = 0.0
                encoded_values[x, y, 2] = 0.0

    return encoded_values


def decode_world_minimap(minimap_data):
    width = minimap_data.shape[0]
    height = minimap_data.shape[1]

    decoded_values = np.empty((width, height, 3), dtype=int)
    for x in range(width):
        for y in range(height):
            decoded_values[x, y, 0] = int(round(minimap_data[x, y, 0] * 255.0))
            decoded_values[x, y, 1] = int(round(minimap_data[x, y, 1] * 255.0))
            decoded_values[x, y, 2] = int(round(minimap_data[x, y, 2] * 255.0))

    return decoded_values


def save_rgb_map(rgb_map, name):
    try:
        width = rgb_map.shape[0]
        height = rgb_map.shape[1]
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        for x in range(width):
            for y in range(height):
                r = rgb_map[x, y, 0]
                g = rgb_map[x, y, 1]
                b = rgb_map[x, y, 2]
                img.putpixel((x, y), (r, g, b))
        img.save(name)
        img.close()
    except:
        print('Failed to save world minimap to %s' % name)


def load_world_data_ver2(world_file):
    world_data_stream = gzip.open(world_file, 'rb')
    world_data = world_data_stream.readlines()
    world_data_stream.close()

    world_width = int(world_data[0].rstrip())
    world_height = int(world_data[1].rstrip())

    layer_size = world_width * world_height

    world = np.zeros((world_width, world_height, 2), dtype=float)

    for z in range(2):
        offset = (z * layer_size) + 2
        for j in range(layer_size):
            x = int(j % world_width)
            y = int(j / world_width)
            world[x, y, z] = int(world_data[offset + j].rstrip())

    return world


def load_world_data_ver3(world_file):
    if not os.path.exists(world_file):
        return None

    world_data_stream = gzip.open(world_file, 'r')
    world_data = world_data_stream.readline().decode('utf8').split(',')
    world_data_stream.close()
    world_width = int(world_data[0].rstrip())
    world_height = int(world_data[1].rstrip())

    layer_size = world_width * world_height

    world = np.zeros((world_width, world_height), dtype=int)

    for j in range(layer_size):
        x = int(j % world_width)
        y = int(j / world_width)
        world[x, y] = int(world_data[2 + j])
    return world


def save_world_data(world_data, name):
    try:
        f = open(name, 'w')
        f.write(str(world_data.shape[0]))
        f.write('\n')
        f.write(str(world_data.shape[1]))
        f.write('\n')
        for y in range(world_data.shape[1]):
            for x in range(world_data.shape[0]):
                f.write(str(int(world_data[x, y])))
                f.write('\n')
        f.close()
    except:
        print('Failed to save world data to %s' % name)


def save_world_minimap(minimap_values, world_data, name):
    if len(world_data.shape) == 2:
        save_world_minimap2d(minimap_values, world_data, name)
    elif len(world_data.shape) == 3 and world_data.shape[2] == 1:
        save_world_minimap2d(minimap_values, np.reshape(world_data, (world_data.shape[0], world_data.shape[1])), name)
    elif len(world_data.shape) == 3 and world_data.shape[2] == 2:
        save_world_minimap3d(minimap_values, world_data, name)
    else:
        print('Unable to save minimap with shape %s' % world_data.shape)


def save_world_minimap2d(minimap, world_data, name):
    try:
        width = world_data.shape[0]
        height = world_data.shape[1]
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        for x in range(width):
            for y in range(height):
                block = int(world_data[x, y])
                if block in minimap:
                    v = minimap[block]
                    r = (v >> 16) & 0xFF
                    g = (v >> 8) & 0xFF
                    b = v & 0xFF
                    img.putpixel((x, y), (r, g, b))
        img.save(name)
        img.close()
    except:
        print('Failed to save world minimap to %s' % name)


def save_world_minimap3d(minimap, world_data, name):
    try:
        width = world_data.shape[0]
        height = world_data.shape[1]

        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        for z in range(2):
            for x in range(width):
                for y in range(height):
                    block = int(world_data[x, y, 1 - z])
                    if block in minimap:
                        v = minimap[block]
                        a = (v >> 24) & 0xFF
                        r = (v >> 16) & 0xFF
                        g = (v >> 8) & 0xFF
                        b = v & 0xFF
                        if a != 0 and b != 0 and v != 0:
                            img.putpixel((x, y), (r, g, b))
        img.save(name)
        img.close()
    except:
        print('Failed to save world minimap to %s' % name)


def save_world_preview(block_images, world_data, name):
    if os.path.exists(name):
        print('%s already exists, skipping.' % name)
        return

    try:
        width = world_data.shape[0]
        height = world_data.shape[1]
        img = Image.new('RGB', (width * 16, height * 16), color=(0, 0, 0))
        for x in range(width):
            for y in range(height):
                block = int(world_data[x, y])
                if block in block_images:
                    block_image = block_images[block]
                    img.paste(block_image, (x * 16, y * 16))
                else:
                    block_image = block_images[0]
                    img.paste(block_image, (x * 16, y * 16))
        img.save(name, compress_level=1)
        img.close()
    except:
        print('Failed to save world minimap to %s' % name)


def shuffle_unison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def check_or_create_local_path(name, base_dir=None):
    if base_dir is None:
        base_dir = os.getcwd()

    local_dir = '%s\\%s\\' % (base_dir, name)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    return local_dir


def delete_folder(path):
    rmtree(path, ignore_errors=True)


def delete_files_in_path(path):
    path_filenames = os.listdir(path)
    for filename in path_filenames:
        full_path = os.path.join(path, filename)
        if os.path.isfile(full_path):
            os.remove(full_path)
        elif os.path.isdir(full_path):
            delete_folder(full_path)


def delete_empty_versions(base_dir, min_files):
    # Loop through versions
    dir_list = os.listdir(base_dir)
    for ver in dir_list:
        if ver != 'graph':
            version_dir = os.path.join(base_dir, ver)
            graph_ver_dir = os.path.join(base_dir, 'graph', ver)

            if not os.path.exists(graph_ver_dir) or len(
                    os.listdir(graph_ver_dir)) < min_files:  # 1 is just model in graph

                # Delete both directories
                delete_folder(version_dir)

                if os.path.exists(graph_ver_dir):
                    delete_folder(graph_ver_dir)


def save_source_to_dir(base_dir):
    source_dir = check_or_create_local_path('src', base_dir)
    cur_dir = os.getcwd()
    for path in os.listdir(cur_dir):
        if os.path.isfile(path):
            copyfile(path, '%s\\%s' % (source_dir, path))


def get_world_id(world_file):
    base = os.path.basename(world_file)
    return os.path.splitext(base)[0]


def get_latest_epoch(directory):
    highest_epoch = -1
    for path in os.listdir(directory):
        fullpath = os.path.join(directory, path)
        if os.path.isdir(fullpath):

            # Check if directory is empty
            isempty = len(os.listdir(fullpath)) == 0
            if isempty:
                continue

            if path[:5] == 'epoch':
                path_epoch = int(path[5:])
                if path_epoch > highest_epoch:
                    highest_epoch = path_epoch

    return highest_epoch


def get_latest_version(directory):
    highest_ver = 0
    for path in os.listdir(directory):
        if path[:3] == 'ver':
            path_ver = int(path[3:])
            if path_ver > highest_ver:
                highest_ver = path_ver

    return highest_ver


def rotate_world90(world_data):
    world_width = world_data.shape[0]
    world_height = world_data.shape[1]

    rotated_world = np.empty((world_width, world_height), dtype=world_data.dtype)
    for y in range(world_height):
        for x in range(world_width):
            cur_id = world_data[x, y]
            rot_id = blocks.rotate_block(cur_id)

            transformed_x = world_height - y - 1
            transformed_y = x

            rotated_world[transformed_x, transformed_y] = rot_id

    return rotated_world


def save_train_data(train_data, block_images, base_dir):
    for i in range(train_data.shape[0]):
        decoded_world = decode_world_sigmoid(train_data[i])
        save_world_preview(block_images, decoded_world, '%s\\image%s.png' % (base_dir, i))


def save_world_repo_previews(world_repo, output_dir):
    block_images = load_block_images()

    cur_dir = os.getcwd()
    repo_dir = '%s\\%s' % (cur_dir, world_repo)
    for world_name in os.listdir(repo_dir):
        world_file = '%s\\%s' % (repo_dir, world_name)
        dest_file = '%s\\%s.png' % (output_dir, world_name)
        if not os.path.exists(dest_file):
            world_data = load_world_data_ver3(world_file)
            if world_data.shape[0] >= 100 and world_data.shape[1] >= 100:
                save_world_preview(block_images, world_data, dest_file)
