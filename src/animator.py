import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.core import Dense, Reshape, Activation
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Input
from keras.models import Model, Sequential
from keras.optimizers import Adam

import utils
from loadworker import load_worlds
from translator import build_translator


def build_animator(size):
    # Takes in latent input, and target minimap colors
    # Outputs a real world whose minimap is supposed to reflect the target minimap

    # Calculate starting kernel size
    n = size
    while n > 7:
        n = n // 2

    # Calculate latent_units, get closest factor with enough units (>1024 units each)
    latent_units = n * n
    while latent_units < 1024:
        latent_units = latent_units * 2

    # Calculate starting filters
    f = (2 * latent_units) // (n * n)

    latent_input = Input(shape=(128,))
    latent = Dense(units=latent_units)(latent_input)

    target_input = Input(shape=(size, size, 3))
    target = Reshape(target_shape=(size * size * 3,))(target_input)
    target = Dense(units=latent_units)(target)

    animator = Concatenate()([latent, target])
    animator = Reshape(target_shape=(n, n, f))(animator)

    s = n
    while s < size:
        animator = Conv2DTranspose(f, kernel_size=5, strides=1, padding='same')(animator)
        animator = BatchNormalization(momentum=0.8)(animator)
        animator = LeakyReLU()(animator)

        animator = Conv2DTranspose(f, kernel_size=3, strides=2, padding='same')(animator)
        animator = BatchNormalization(momentum=0.8)(animator)
        animator = LeakyReLU()(animator)

        s = s * 2
        f = f // 2

    animator = Conv2DTranspose(10, kernel_size=5, strides=1, padding='same')(animator)
    animator = Activation('sigmoid')(animator)

    animator_model = Model(inputs=[latent_input, target_input], outputs=animator)
    animator_model.summary()

    translator_model = build_translator(size)
    translator_model.trainable = False

    animator_trainer_model = Sequential()
    animator_trainer_model.add(animator_model)
    animator_trainer_model.add(translator_model)
    animator_trainer_model.summary()

    return animator_trainer_model, animator_model


def train(epochs, batch_size, world_count, version_name=None, initial_epoch=0):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    all_models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path('animator', all_models_dir)

    utils.delete_empty_versions(model_dir, 1)
    no_version = version_name is None
    if no_version:
        latest = utils.get_latest_version(model_dir)
        version_name = 'ver%s' % (latest + 1)

    version_dir = utils.check_or_create_local_path(version_name, model_dir)
    graph_dir = utils.check_or_create_local_path('graph', model_dir)
    graph_version_dir = utils.check_or_create_local_path(version_name, graph_dir)

    previews_dir = utils.check_or_create_local_path('previews', version_dir)
    model_save_dir = utils.check_or_create_local_path('models', version_dir)

    print('Saving source...')
    utils.save_source_to_dir(version_dir)

    print('Loading minimap values...')
    minimap_values = utils.load_minimap_values(res_dir)

    print('Loading block images...')
    block_images = utils.load_block_images(res_dir)

    print('Loading encoding dictionaries...')
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'blocks_colored')

    print('Building model from scratch...')
    optim = Adam(lr=0.0001)

    animator, c_input = build_animator(64)

    animator.summary()
    animator.compile(loss='mse', optimizer=optim)

    print('Loading worlds...')
    x_train = load_worlds(world_count, '%s\\worlds\\' % res_dir, (112, 112), block_forward)
    y_none = np.zeros((batch_size, 112, 112, 10))

    world_count = x_train.shape[0]
    number_of_batches = (world_count - (world_count % batch_size)) // batch_size

    # Initialize tables for Hashtable tensors
    K.get_session().run(tf.tables_initializer())

    for epoch in range(initial_epoch, epochs):

        # Create directories for current epoch
        cur_previews_dir = utils.check_or_create_local_path('epoch%s' % epoch, previews_dir)
        cur_models_dir = utils.check_or_create_local_path('epoch%s' % epoch, model_save_dir)

        print('Shuffling data...')
        np.random.shuffle(x_train)

        for minibatch_index in range(number_of_batches):
            worlds = x_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
            minimaps = get_minimaps(worlds, block_backward, minimap_values)

            if minibatch_index == number_of_batches - 1:
                generated = animator.predict(minimaps)

                for batchImage in range(batch_size):
                    generated_world = generated[batchImage]
                    decoded_generated = utils.decode_world_sigmoid(block_backward, generated_world)
                    utils.save_world_preview(block_images, decoded_generated,
                                             '%s\\generated%s.png' % (cur_previews_dir, batchImage))

                    decoded_world = utils.decode_world_sigmoid(block_backward, worlds[batchImage])
                    utils.save_world_minimap(minimap_values, decoded_world,
                                             '%s\\target%s.png' % (cur_previews_dir, batchImage))

            loss = animator.train_on_batch(minimaps, minimaps)
            print('epoch = %s, loss = %s' % (epoch, loss))


def main():
    train(epochs=13, batch_size=32, world_count=1000)
    # predict('ver9', dict_src_name='pro_labels')


if __name__ == '__main__':
    main()
