import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Input
from keras.optimizers import Adam

import utils
from loadworker import load_worlds


def build_animator(size):
    input = Input(shape=(size, size, 3))

    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(input)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=5, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=128, kernel_size=5, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=10, kernel_size=5, strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('sigmoid')(x)

    animator_model = Model(inputs=[input], outputs=[x])
    return animator_model, input


def get_minimaps(worlds, block_backward, minimap_values):
    # Get minimaps
    batch_size = worlds.shape[0]
    world_width = worlds.shape[1]
    world_height = worlds.shape[2]
    minimaps = np.zeros((batch_size, world_width, world_height, 3), dtype=float)

    for i in range(batch_size):
        decoded_world = utils.decode_world_sigmoid(block_backward, worlds[i])
        minimaps[i] = utils.encode_world_minimap2d(minimap_values, decoded_world)

    return minimaps


def painting_loss(minimap_values):
    def loss(y_true, y_pred):
        output_worlds = K.eval(y_pred)
        output_pred = get_minimaps(output_worlds, minimap_values)
        output_tensor = tf.convert_to_tensor(output_pred)

        input_worlds = K.eval(y_true)

        input_true = get_minimaps(input_worlds, minimap_values)
        input_tensor = tf.convert_to_tensor(input_true)

        # return K.abs(input_tensor - output_tensor)
        return K.mean(K.square(output_tensor - input_tensor), axis=-1)

    return loss


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

    previews_dir = utils.check_or_create_local_path("previews", version_dir)
    model_save_dir = utils.check_or_create_local_path("models", version_dir)

    print("Saving source...")
    utils.save_source_to_dir(version_dir)

    print("Loading minimap values...")
    minimap_values = utils.load_minimap_values(res_dir)

    print("Loading encoding dictionaries...")
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'optimized')

    print("Building model from scratch...")
    optim = Adam(lr=0.0001)

    animator, c_input = build_animator(112)

    animator.summary()
    animator.compile(loss='mse', optimizer=optim)

    print("Loading worlds...")
    x_train = load_worlds(world_count, '%s\\worlds\\' % res_dir, (112, 112), block_forward)
    y_none = np.zeros((batch_size, 112, 112, 10))

    world_count = x_train.shape[0]
    number_of_batches = (world_count - (world_count % batch_size)) // batch_size

    for epoch in range(initial_epoch, epochs):

        # Create directories for current epoch
        cur_previews_dir = utils.check_or_create_local_path("epoch%s" % epoch, previews_dir)
        cur_models_dir = utils.check_or_create_local_path("epoch%s" % epoch, model_save_dir)

        print("Shuffling data...")
        np.random.shuffle(x_train)

        for minibatch_index in range(number_of_batches):
            worlds = x_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
            minimaps = get_minimaps(worlds, block_backward, minimap_values)

            loss = animator.train_on_batch(minimaps, worlds)
            print("epoch = %s, loss = %s" % (epoch, loss))


def main():
    train(epochs=13, batch_size=32, world_count=1000)
    # predict('ver9', dict_src_name='pro_labels')


if __name__ == "__main__":
    main()
