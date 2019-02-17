import math
import os

import keras
import numpy as np
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers.core import Dense, Reshape, Activation
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Input
from keras.models import Model, Sequential, load_model

import utils
from loadworker import load_worlds_with_minimaps
from pro_classifier import build_classifier


def build_pro_animator(size):
    # Takes in latent input, and target minimap colors
    # Outputs a pro looking world whose minimap also reflects the target minimap

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

    # Build model to train animator to produce maps with the correct minimap
    translator_model = build_translator(size)
    translator_model.trainable = False
    # TODO: Load translator model

    animator_minimap_model = translator_model(animator_model)

    # Build model to train animator to produce good looking maps
    classifier_model = build_classifier(size)
    classifier_model.trainable = False
    # TODO: Load pro model

    animator_pro_model = Sequential()
    animator_pro_model.add(animator_model)
    animator_pro_model.add(classifier_model)
    animator_pro_model.summary()

    return animator_minimap_model, animator_pro_model, animator_model


def build_basic_animator(size):
    # Takes in latent input, and target minimap colors
    # Outputs a real world whose minimap is supposed to reflect the target minimap

    animator_input = Input(shape=(size, size, 3))
    animator = Conv2D(filters=2048, kernel_size=(1, 1), strides=(1, 1), padding='same')(animator_input)
    # animator = BatchNormalization(momentum=0.8, axis=3)(animator)
    animator = Activation('relu')(animator)
    animator = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same')(animator)
    animator = Activation('sigmoid')(animator)

    animator_model = Model(inputs=animator_input, outputs=animator)
    animator_model.summary()
    animator_model.trainable = True

    return animator_model


def train(epochs, batch_size, world_count, sz=64, version_name=None):
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
    mm_values = utils.load_minimap_values(res_dir)

    print('Loading block images...')
    block_images = utils.load_block_images(res_dir)

    print('Loading encoding dictionaries...')
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'blocks_colored')

    print('Building model from scratch...')
    animator = build_basic_animator(112)
    animator.compile(loss='mse', optimizer='adam')

    translator = load_model('%s\\translator\\ver15\\models\\best_loss.h5' % all_models_dir)
    translator.trainable = False

    animator_minimap = Sequential()
    animator_minimap.add(animator)
    animator_minimap.add(translator)
    animator_minimap.compile(loss='mse', optimizer='adam')

    print('Saving model images...')
    keras.utils.plot_model(animator, to_file='%s\\animator.png' % version_dir, show_shapes=True, show_layer_names=True)

    # Set up tensorboard
    print('Setting up tensorboard...')
    tb_callback = keras.callbacks.TensorBoard(log_dir=graph_version_dir, write_graph=True)
    tb_callback.set_model(animator_minimap)

    # before training init writer (for tensorboard log) / model
    tb_writer = tf.summary.FileWriter(logdir=graph_version_dir)
    mm_loss = tf.Summary()
    mm_loss.value.add(tag='ae_loss', simple_value=None)

    print('Loading worlds...')
    y_train, x_train = load_worlds_with_minimaps(world_count, '%s\\worlds\\' % res_dir, (sz, sz), block_forward,
                                                 mm_values)

    world_count = x_train.shape[0]
    number_of_batches = (world_count - (world_count % batch_size)) // batch_size

    for epoch in range(epochs):

        # Create directories for current epoch
        cur_previews_dir = utils.check_or_create_local_path('epoch%s' % epoch, previews_dir)
        cur_models_dir = utils.check_or_create_local_path('epoch%s' % epoch, model_save_dir)

        print('Shuffling data...')
        np.random.shuffle(x_train)

        for minibatch_index in range(number_of_batches):
            minimaps = x_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
            actual = y_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
            if minibatch_index == number_of_batches - 1:
                # Save previews
                print('Saving previews...')
                worlds = animator.predict(minimaps)
                for i in range(batch_size):
                    world_decoded = utils.decode_world_sigmoid(block_backward, worlds[i])
                    utils.save_world_preview(block_images, world_decoded, '%s\\animated%s.png' % (cur_previews_dir, i))

                    mm_decoded = utils.decode_world_minimap(minimaps[i])
                    utils.save_rgb_map(mm_decoded, '%s\\target%s.png' % (cur_previews_dir, i))

            # Train animator
            # world_loss = animator.train_on_batch(minimaps, actual)
            minimap_loss = animator_minimap.train_on_batch(minimaps, minimaps)

            # Write loss
            if not math.isnan(minimap_loss):
                mm_loss.value[0].simple_value = minimap_loss
                tb_writer.add_summary(mm_loss, (epoch * number_of_batches) + minibatch_index)

            print(f"Epoch = {epoch}/{epochs} :: Batch = {minibatch_index}/{number_of_batches} "
                  f":: MMLoss = {minimap_loss}")

            # Save models
            if minibatch_index % 100 == 99 or minibatch_index == number_of_batches - 1:
                try:
                    animator.save('%s\\animator.h5' % cur_models_dir)
                    animator.save_weights('%s\\animator.weights' % cur_models_dir)
                except ImportError:
                    print('Failed to save data.')


def main():
    train(epochs=13, batch_size=10, world_count=10000, sz=112)
    # predict('ver9', dict_src_name='pro_labels')


if __name__ == '__main__':
    main()
