import gzip
import math
import os
from random import randint
from shutil import copyfile

import numpy as np
from PIL import Image
from skimage.draw import ellipse
from skimage.draw import line

import blocks


def load_minimap_values():
    minimap_dict = {}
    with open("./block_colors.txt") as fp:
        line = fp.readline()
        while line:
            space_index = line.index(" ")
            key_str = line[:space_index]
            argb_str = line[space_index:]
            key = int(key_str)
            argb = int(argb_str)
            minimap_dict[key] = argb
            line = fp.readline()
    return minimap_dict


def load_block_images():
    block_dict = {}
    for filename in os.listdir("./blocks"):
        file = "./blocks/%s" % filename
        if os.path.isfile(file):
            block_dict[int(filename[1:-4])] = Image.open(file)
    return block_dict


def load_encoding_dict(name):
    block_forward_dict = [{}, {}, {}, {}]
    block_backward_dict = [{}, {}, {}, {}]
    with open("./%s.txt" % name) as fp:
        line = fp.readline()
        while line:
            block = int(line)
            block_category = 0  # blocks.get_block_category(block)
            assert block not in block_forward_dict
            block_hash = len(block_forward_dict[block_category])
            block_forward_dict[block_category][block] = block_hash
            block_backward_dict[block_category][block_hash] = block
            line = fp.readline()
    assert len(block_forward_dict) == len(block_backward_dict)
    return block_forward_dict, block_backward_dict


def trapez(y, y0, w):
    return np.clip(np.minimum(y + 1 + w / 2 - y0, -y + 1 + w / 2 + y0), 0, 1)


def random_mask(height, width, channels=11):
    """Generates a random irregular mask based off of rectangles"""
    mask = np.zeros((height, width, channels), np.int8)
    img = np.zeros((height, width, 3), np.int8)

    # Random rectangles
    for i in range(randint(1, 2)):
        x1 = randint(1, width - 25)
        x_width = randint(5, min(24, width - x1))
        x2 = x1 + x_width

        y1 = randint(1, height - 25)
        y_height = randint(5, min(24, height - y1))
        y2 = y1 + y_height

        mask[x1:x2, y1:y2, :] = 1

    # Random circles
    for j in range(randint(1, 2)):
        x1 = randint(1, width - 25)
        x_width = randint(5, min(24, width - x1))
        y1 = randint(1, height - 25)
        y_height = randint(5, min(24, height - y1))
        rr, cc = ellipse(y1, x1, y_height, x_width)
        mask[cc, rr, :] = 1

    # Random lines
    for j in range(randint(10, 20)):
        x1, x2 = randint(1, width - 1), randint(1, width - 1)
        y1, y2 = randint(1, height - 1), randint(1, height - 1)
        rr, cc = line(y1, x1, y2, x2)
        mask[cc, rr, :] = 1

    return 1 - mask


def mask_batch(batch):
    masked = np.empty(batch.shape, dtype=np.int8)
    masks = np.empty(batch.shape, dtype=np.int8)

    batch_size = batch.shape[0]
    for i in range(batch_size):
        world_masked = np.copy(batch[i])
        mask = random_mask(world_masked.shape[0], world_masked.shape[1], world_masked.shape[2])
        world_masked[mask == 0] = 1
        masked[i] = world_masked
        masks[i] = mask

    return [masked, masks]


def decode_world2d(block_backward_dict, world_data):
    width = world_data.shape[0]
    height = world_data.shape[1]
    world_copy = np.zeros((width, height), dtype=int)
    for y in range(height):
        for x in range(width):
            encoded_value = world_data[x, y, 0]
            world_copy[x, y] = decode_block(block_backward_dict, encoded_value, 0)
    return world_copy


def decode_world2d_binary(world_data):
    width = world_data.shape[0]
    height = world_data.shape[1]
    world_copy = np.zeros((width, height), dtype=int)
    for y in range(height):
        for x in range(width):
            value = 0
            for bit in range(world_data.shape[2]):
                bit_data = world_data[x, y, bit]
                bit_value = 0
                if bit_data >= 0.5:
                    bit_value = 1
                value = value | (bit_value << (10 - bit))
            world_copy[x, y] = int(value)
    return world_copy


def decode_block(block_backward_dict, encoded_value, layer):
    if not np.isscalar(encoded_value):
        encoded_value = encoded_value[0]

    category_max = len(block_backward_dict[layer]) - 1
    hash_decimal = (((encoded_value + 1) / 2) * category_max)
    truncated_hash = math.floor(hash_decimal)
    if truncated_hash in block_backward_dict[layer]:
        block_id = block_backward_dict[layer][truncated_hash]
        return block_id
    else:
        # print("Decode not in list, value is %s" % truncatedHash)
        return 0


def encode_world2d(block_forward_dict, world_data):
    width = world_data.shape[0]
    height = world_data.shape[1]
    world_copy = np.zeros((width, height, 1), dtype=float)

    if len(world_data.shape) == 2:
        for y in range(height):
            for x in range(width):
                layer, value = encode_block(block_forward_dict, world_data[x, y])
                # world_copy[x, y, 0] = (layer / 2) - 1
                world_copy[x, y, 0] = value
    elif len(world_data.shape) == 3:  # Just take foreground
        for y in range(height):
            for x in range(width):
                layer, value = encode_block(block_forward_dict, world_data[x, y, 0])
                # world_copy[x, y, 0] = (layer / 2) - 1
                world_copy[x, y, 0] = value

    return world_copy


def encode_world2d_binary(world_data):
    width = world_data.shape[0]
    height = world_data.shape[1]
    world_copy = np.zeros((width, height, 11), dtype=np.int8)

    if len(world_data.shape) == 2:
        for y in range(height):
            for x in range(width):
                value = int(world_data[x, y])
                for bit in range(11):
                    bit_value = (value >> bit) & 1
                    bit_value_reshaped = bit_value  # (bit_value * 2) - 1
                    world_copy[x, y, 10 - bit] = bit_value_reshaped  # [-1, 1]
    elif len(world_data.shape) == 3:  # Just take foreground
        for y in range(height):
            for x in range(width):
                value = int(world_data[x, y, 0])
                for bit in range(11):
                    bit_value = (value >> bit) & 1
                    bit_value_reshaped = bit_value  # (bit_value * 2) - 1
                    world_copy[x, y, 10 - bit] = bit_value_reshaped  # [-1, 1]

    return world_copy


def encode_block(block_forward_dict, block_id):
    block_category = 0  # blocks.get_block_category(block_id)
    if block_id not in block_forward_dict[block_category]:
        return encode_block(block_forward_dict, 0)

    category_max = len(block_forward_dict[block_category]) - 1
    return block_category, ((block_forward_dict[block_category][block_id] / category_max) * 2) - 1  # [-1, 1] scaling


def encode_block_color(minimap_values, block):
    v = minimap_values[block]
    a = (v >> 24) & 0xFF
    r = (v >> 16) & 0xFF
    g = (v >> 8) & 0xFF
    b = v & 0xFF
    if a != 0 and b != 0 and v != 0:
        return (r - 127.5) / 127.5, (g - 127.5) / 127.5, (b - 127.5) / 127.5


def encode_world_minimap(minimap_values, world_data):
    if len(world_data.shape) == 2:
        return encode_world_minimap2d(minimap_values, world_data)
    elif len(world_data.shape) == 3 and world_data.shape[2] == 1:
        return encode_world_minimap2d(minimap_values,
                                      np.reshape(world_data, (world_data.shape[0], world_data.shape[1])))
    elif len(world_data.shape) == 3 and world_data.shape[2] == 2:
        return encode_world_minimap3d(minimap_values, world_data)
    else:
        print("Unable to encode world minimap with shape %s" % world_data.shape)


def encode_world_minimap2d(minimap_values, world_data):
    width = world_data.shape[0]
    height = world_data.shape[1]

    encoded_values = np.zeros((width, height, 3), dtype=float)
    for x in range(width):
        for y in range(height):
            block = int(world_data[x, y])
            if block in minimap_values:
                v = minimap_values[block]
                a = (v >> 24) & 0xFF
                r = (v >> 16) & 0xFF
                g = (v >> 8) & 0xFF
                b = v & 0xFF
                if a != 0 and b != 0 and v != 0:
                    encoded_values[x, y, 0] = (r - 127.5) / 127.5
                    encoded_values[x, y, 1] = (g - 127.5) / 127.5
                    encoded_values[x, y, 2] = (b - 127.5) / 127.5

    return encoded_values


def encode_world_minimap3d(minimap_values, world_data):
    width = world_data.shape[0]
    height = world_data.shape[1]

    encoded_values = np.zeros((width, height, 3), dtype=float)
    for z in range(2):
        for x in range(width):
            for y in range(height):
                block = int(world_data[x, y, 1 - z])
                if block in minimap_values:
                    v = minimap_values[block]
                    a = (v >> 24) & 0xFF
                    r = (v >> 16) & 0xFF
                    g = (v >> 8) & 0xFF
                    b = v & 0xFF
                    if a != 0 and b != 0 and v != 0:
                        encoded_values[x, y, 0] = (r - 127.5) / 127.5
                        encoded_values[x, y, 1] = (g - 127.5) / 127.5
                        encoded_values[x, y, 2] = (b - 127.5) / 127.5

    return encoded_values


def decode_world_minimap(world_data):
    width = world_data.shape[0]
    height = world_data.shape[1]

    decoded_values = np.zeros((width, height, 3), dtype=int)
    for x in range(width):
        for y in range(height):
            decoded_values[x, y, 0] = (world_data[x, y, 0] * 127.5) + 127.5
            decoded_values[x, y, 1] = (world_data[x, y, 1] * 127.5) + 127.5
            decoded_values[x, y, 2] = (world_data[x, y, 2] * 127.5) + 127.5

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
        print("Failed to save world minimap to %s" % name)


def load_world_data_ver2(world_file):
    world_data_stream = gzip.open(world_file, "rb")
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
    world_data_stream = gzip.open(world_file, "r")
    world_data = world_data_stream.readline().decode("utf8").split(',')
    world_data_stream.close()
    world_width = int(world_data[0].rstrip())
    world_height = int(world_data[1].rstrip())

    layer_size = world_width * world_height

    world = np.zeros((world_width, world_height), dtype=float)

    for j in range(layer_size):
        x = int(j % world_width)
        y = int(j / world_width)
        world[x, y] = int(world_data[2 + j])
    return world


def save_world_data(world_data, name):
    try:
        f = open(name, "w")
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
        print("Failed to save world data to %s" % name)


def save_world_minimap(minimap, world_data, name):
    if len(world_data.shape) == 2:
        save_world_minimap2d(minimap, world_data, name)
    elif len(world_data.shape) == 3 and world_data.shape[2] == 1:
        save_world_minimap2d(minimap, np.reshape(world_data, (world_data.shape[0], world_data.shape[1])), name)
    elif len(world_data.shape) == 3 and world_data.shape[2] == 2:
        save_world_minimap3d(minimap, world_data, name)
    else:
        print("Unable to save minimap with shape %s" % world_data.shape)


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
        print("Failed to save world minimap to %s" % name)


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
        print("Failed to save world minimap to %s" % name)


def save_world_preview(block_images, world_data, name):
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
        print("Failed to save world minimap to %s" % name)


def shuffle_unison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def check_or_create_local_path(name, base_dir=None):
    if base_dir is None:
        base_dir = os.getcwd()

    local_dir = "%s\\%s\\" % (base_dir, name)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    return local_dir


def delete_files_in_path(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except:
            pass


def save_source_to_dir(base_dir):
    source_dir = check_or_create_local_path("src", base_dir)
    cur_dir = os.getcwd()
    for path in os.listdir(cur_dir):
        if os.path.isfile(path):
            copyfile(path, "%s\\%s" % (source_dir, path))


def get_latest_version(directory):
    highest_ver = 0
    for path in os.listdir(directory):
        path_ver = int(path[3:])
        if path_ver > highest_ver:
            highest_ver = path_ver

    return highest_ver


def rotate_world90(world_data):
    world_width = world_data.shape[0]
    world_height = world_data.shape[1]

    rotated_world = np.zeros((world_width, world_height), dtype=world_data.dtype)
    for y in range(world_height):
        for x in range(world_width):
            cur_id = world_data[x, y]
            rot_id = blocks.rotate_block(cur_id)

            transformed_x = world_height - y - 1
            transformed_y = x

            rotated_world[transformed_x, transformed_y] = rot_id

    return rotated_world


def save_train_data(train_data, block_images, dir):
    for i in range(train_data.shape[0]):
        decoded_world = decode_world2d_binary(train_data[i])
        save_world_preview(block_images, decoded_world, '%s\\image%s.png' % (dir, i))


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
