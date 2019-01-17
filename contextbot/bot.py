# Import the library
import os
import time
from threading import Lock

import numpy as np
import tensorflow as tf

import ae
import contextbot.initparse
import unet
import utils
from playerio import *

world_data = None  # block data for bot
pconv_unet = None
block_images = utils.load_block_images()
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
    for i in range(len(build_queue) - 1, 0, -1):
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

    world_x1 = max(0, avg_x - 32)
    world_x2 = min(world_data.shape[1], world_x1 + 64)

    world_y1 = max(0, avg_y - 32)
    world_y2 = min(world_data.shape[1], world_y1 + 64)

    input_mask = np.ones((64, 64, 11), dtype=np.int8)
    for i in range(len(coords)):
        loc_x = coords[i][0] - world_x1
        loc_y = coords[i][1] - world_y2
        input_mask[loc_x, loc_y, :] = 0

    input_data = np.zeros((64, 64), dtype=int)
    for x in range(world_x1, world_x2):
        for y in range(world_y1, world_y2):
            loc_x = x - world_x1
            loc_y = y - world_y1
            cur_block_data = world_data[x, y, 0]
            input_data[loc_x, loc_y] = cur_block_data.Id

    encoded_input = utils.encode_world2d_binary(input_data)
    encoded_input[input_mask == 0] = 1

    encoded_context_data = None
    with graph.as_default():
        encoded_context_data = pconv_unet.predict([[encoded_input], [input_mask]])

    context_data = utils.decode_world2d_binary(encoded_context_data[0])
    utils.save_world_preview(block_images, context_data, '%s\\real.png' % cur_dir)

    for x in range(64):
        for y in range(64):
            if input_mask[x, y, 0] == 0:
                block_id = int(context_data[x, y])
                world_x = x + world_x1
                world_y = y + world_y1
                if world_data[world_x, world_y, 0].Id != block_id:
                    r.send('b', 0, world_x, world_y, block_id)
                    time.sleep(25 / 1000.0)


@EventHandler.add('init')
def on_init(r, init_message):
    print("Joined.")
    r.send('init2')

    global world_data
    world_data = contextbot.initparse.get_world_data(init_message)

    wd = np.zeros((200, 400), dtype=int)
    for x in range(200):
        for y in range(400):
            wd[x, y] = world_data[x, y, 0].Id

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
        world_data[block_x, block_y, 0].Id = block_id

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
print("Logging in...")
username = None
password = None
with open("./ugp") as fp:
    line = fp.readline()
    spl = line.split(' ')
    username = spl[0]
    password = spl[1]

client = Client('everybody-edits-su9rn58o40itdbnw69plyw', username, password)

# Get the game version from BigDB
version = client.bigdb_load('config', 'config')['version']

# Join a room

bot_room = client.create_join_room('PW-48NLanscEI', f'Everybodyedits{version}', True)

# Send a message
print("Joining world...")
bot_room.send('init')

build_queue = []  # (x, y) mask queue for bot

cur_dir = 'C:\\Users\\austi\\Documents\\PycharmProjects\\KerasEE\\'

print("Loading feature model...")
feature_model = ae.autoencoder_model()
feature_model.load_weights('%s\\ae\\ver5\\models\\epoch38\\autoencoder.weights' % cur_dir)
feature_layers = [7, 14, 21]

print("Loading context model...")
contextnet = unet.PConvUnet(feature_model, feature_layers, width=64, height=64, inference_only=False)
pconv_unet = contextnet.build_pconv_unet(train_bn=False, lr=0.0001)
pconv_unet.load_weights('%s\\contextnet\\ver52\\models\\epoch5\\unet.weights' % cur_dir)
graph = tf.get_default_graph()

print("Done loading model.")
init_lock = Lock()
init_lock.acquire()
print("Ready...")
init_lock.acquire()
