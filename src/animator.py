import os

import numpy as np
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

import utils
from loadworker import load_worlds


def build_animator(size):
    model = Sequential()

    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', input_shape=(size, size, 3)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', input_shape=(size, size, 3)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=10, kernel_size=5, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('sigmoid'))

    model.add(Conv2D(filters=3, kernel_size=5, strides=1, padding='same', trainable=False))

    return model


def get_minimaps(worlds, minimap_values):
    # Get minimaps
    batch_size = worlds.shape[0]
    world_width = worlds.shape[1]
    world_height = worlds.shape[2]
    minimaps = np.zeros((batch_size, world_width, world_height), dtype=float)

    for i in range(batch_size):
        minimaps[i] = utils.encode_world_minimap2d(minimap_values, worlds[i])

    return minimaps


def train(epochs, batch_size, world_count, version_name=None, initial_epoch=0):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    all_models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path("animator", all_models_dir)

    utils.delete_empty_versions(model_dir, 1)
    no_version = version_name is None
    if no_version:
        latest = utils.get_latest_version(model_dir)
        version_name = "ver%s" % (latest + 1)

    version_dir = utils.check_or_create_local_path(version_name, model_dir)
    graph_dir = utils.check_or_create_local_path("graph", model_dir)
    graph_version_dir = utils.check_or_create_local_path(version_name, graph_dir)

    model_save_dir = utils.check_or_create_local_path("models", version_dir)

    print("Saving source...")
    utils.save_source_to_dir(version_dir)

    print("Loading encoding dictionaries...")
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'optimized')

    print("Building model from scratch...")
    optim = Adam(lr=0.0001)

    c = build_animator(112)

    c.summary()
    c.compile(loss="mse", optimizer=optim)

    print("Loading worlds...")
    x_train = load_worlds(world_count, '%s\\worlds\\' % res_dir, (112, 112), block_forward)


def main():


# train(epochs=13, batch_size=100, world_count=25000, dict_src_name='pro_labels')
# predict('ver9', dict_src_name='pro_labels')


if __name__ == "__main__":
    main()
