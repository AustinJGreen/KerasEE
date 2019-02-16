# Import the library
import os
import time
from threading import Lock

import numpy as np
import tensorflow as tf

import unet_model
import utils
from playerio import *
from playerio.initparse import get_world_data

world_data = None  # block data for bot
pconv_unet = None
block_images = utils.load_block_images('C:\\Users\\austi\\Documents\\PycharmProjects\\KerasEE\\res\\')
block_forward, block_backward = utils.load_encoding_dict('C:\\Users\\austi\\Documents\\PycharmProjects\\KerasEE\\res\\',
                                                         'blocks_optimized')
global graph


def build_for(r, player_id):
    global world_data
    global pconv_unet
    global block_images

    coords = []
    min_x = 1000
    max_x = -1
    min_y = 1000
    max_y = -1
    total_x = 0
    total_y = 0
    count = 0
    for i in range(len(build_queue) - 1, -1, -1):
        if build_queue[i][0] == player_id or True:
            x_coord = build_queue[i][1]
            y_coord = build_queue[i][2]
            min_x = min(x_coord, min_x)
            max_x = max(x_coord, max_x)
            min_y = min(y_coord, min_y)
            max_y = max(y_coord, max_y)
            total_x += x_coord
            total_y += y_coord
            count += 1
            coords.append((x_coord, y_coord))
            del build_queue[i]

    # For now take average location and get context from around this center
    avg_x = total_x // count
    avg_y = total_y // count

    world_x1 = max(0, avg_x - 64)
    world_x2 = min(world_data.shape[1] - 1, world_x1 + 128)

    world_y1 = max(0, avg_y - 64)
    world_y2 = min(world_data.shape[1] - 1, world_y1 + 128)

    input_mask = np.ones((128, 128, 10), dtype=np.int8)
    for i in range(len(coords)):
        loc_x = coords[i][0] - world_x1
        loc_y = coords[i][1] - world_y2
        input_mask[loc_x, loc_y, :] = 0

    input_data = np.zeros((128, 128), dtype=int)
    for x in range(world_x1, world_x2):
        for y in range(world_y1, world_y2):
            loc_x = x - world_x1
            loc_y = y - world_y1
            cur_block_data = world_data[x, y, 0]
            input_data[loc_x, loc_y] = cur_block_data.block_id

    utils.save_world_preview(block_images, input_data, '%s\\input.png' % cur_dir)

    encoded_input = utils.encode_world_sigmoid(block_forward, input_data)
    encoded_input[input_mask == 0] = 1

    encoded_context_data = None
    with graph.as_default():
        encoded_context_data = pconv_unet.predict([[encoded_input], [input_mask]])

    context_data = utils.decode_world_sigmoid(block_backward, encoded_context_data[0])
    utils.save_world_preview(block_images, context_data, '%s\\real.png' % cur_dir)

    for i in range(len(coords)):
        x = coords[i][0]
        y = coords[i][1]
        loc_x = x - world_x1
        loc_y = y - world_y1
        block_id = int(context_data[loc_x, loc_y])
        if world_data[x, y, 0].block_id != block_id:
            r.send('b', 0, x, y, block_id)
            time.sleep(25 / 1000.0)


@EventHandler.add('init')
def on_init(r, init_message):
    print('Joined.')
    r.send('init2')

    width = init_message[18]
    height = init_message[19]

    global world_data
    world_data = get_world_data(init_message)

    wd = np.empty((width, height), dtype=int)
    for x in range(width):
        for y in range(height):
            wd[x, y] = world_data[x, y, 0].block_id

    utils.save_world_preview(block_images, wd, '%s\\init.png' % os.getcwd())


@EventHandler.add('b')
def on_block(r, b_message):
    layer_id = b_message[0]
    block_x = b_message[1]
    block_y = b_message[2]
    block_id = b_message[3]
    player_id = b_message[4]

    global world_data

    if layer_id == 0:
        world_data[block_x, block_y, 0].block_id = block_id

    if block_id == 12:
        build_queue.append((player_id, block_x, block_y))


@EventHandler.add('br')
def on_block(r, b_message):
    block_x = b_message[0]
    block_y = b_message[1]
    block_id = b_message[2]
    world_data[block_x, block_y] = block_id


@EventHandler.add('say')
def on_say(r, say_message):
    player_id = say_message[0]
    text = say_message[1]
    if text == '!build':
        build_for(r, player_id)
    elif text == '!quit':
        init_lock.release()


@EventHandler.add('playerio.disconnect')
def on_disconnect(r, disconnect_message):
    print('Disconnected :(')


# Connect to the game
print('Logging in...')
username = None
password = None
with open('C:\\Users\\austi\\Documents\\PycharmProjects\\KerasEE\\res\\ugp') as fp:
    line = fp.readline()
    spl = line.split(' ')
    username = spl[0]
    password = spl[1]

client = Client('everybody-edits-su9rn58o40itdbnw69plyw', username, password)

# Get the game version from BigDB
version = client.bigdb_load('config', 'config')['version']

# Join a room

bot_room = client.create_join_room('PWrO5qmOGjb0I', f'Everybodyedits{version}', True)

# Send a message
print('Joining world...')
bot_room.send('init')

build_queue = []  # (x, y) mask queue for bot

cur_dir = 'C:\\Users\\austi\\Documents\\PycharmProjects\\KerasEE\\'

print('Loading context model...')
contextnet = unet_model.PConvUnet(None, [7, 14, 21], inference_only=True)
pconv_unet = contextnet.build_pconv_unet(train_bn=False, lr=0.0001)
pconv_unet.load_weights('%s\\models\\inpainting\\ver8\\models\\epoch3\\unet.weights' % cur_dir)
graph = tf.get_default_graph()

print('Done loading model.')
init_lock = Lock()
init_lock.acquire()
print('Ready...')
init_lock.acquire()
