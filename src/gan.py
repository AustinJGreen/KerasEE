import os
import time

import keras
import numpy as np
from keras.layers import Dense, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam

import utils
from loadworker import load_worlds_with_label
from tbmanager import TensorboardManager


def build_generator(size):
    model = Sequential(name='generator')

    # Calculate starting kernel size
    n = size
    while n > 7:
        n = n // 2

    s = n
    f = 512

    model.add(Dense(input_dim=256, units=n * n * f))
    model.add(LeakyReLU())

    model.add(Reshape((n, n, f)))

    while s < size:
        model.add(Conv2DTranspose(f, kernel_size=5, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(f, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU())

        s = s * 2
        f = f // 2

    model.add(Conv2DTranspose(10, kernel_size=5, strides=1, padding='same'))
    model.add(Activation('sigmoid'))

    return model


def build_discriminator(size):
    model = Sequential(name='discriminator')

    f = 64
    s = size

    while s > 7:
        if s == size:
            model.add(Conv2D(filters=f, kernel_size=7, strides=1, padding='same', input_shape=(size, size, 10)))
        else:
            model.add(Conv2D(filters=f, kernel_size=7, strides=1, padding='same'))

        model.add(BatchNormalization(momentum=0.8, axis=3))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=f, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.8, axis=3))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(SpatialDropout2D(0.2))

        f = f * 2
        s = s // 2

    model.add(GlobalAveragePooling2D())

    model.add(Dense(1, activation='sigmoid'))

    model.trainable = True
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def train(epochs, batch_size, world_count, version_name=None, initial_epoch=0):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    all_models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path('gan', all_models_dir)

    utils.delete_empty_versions(model_dir, 1)
    no_version = version_name is None
    if no_version:
        latest = utils.get_latest_version(model_dir)
        version_name = f'ver{latest + 1}'

    version_dir = utils.check_or_create_local_path(version_name, model_dir)
    graph_dir = utils.check_or_create_local_path('graph', model_dir)
    graph_version_dir = utils.check_or_create_local_path(version_name, graph_dir)

    worlds_dir = utils.check_or_create_local_path('worlds', version_dir)
    previews_dir = utils.check_or_create_local_path('previews', version_dir)
    model_save_dir = utils.check_or_create_local_path('models', version_dir)

    print('Saving source...')
    utils.save_source_to_dir(version_dir)

    print('Loading block images...')
    block_images = utils.load_block_images(res_dir)

    print('Loading encoding dictionaries...')
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'blocks_optimized')

    # Load model and existing weights
    print('Loading model...')

    # Try to load full model, otherwise try to load weights
    size = 64
    cur_models = f'{model_save_dir}\\epoch{initial_epoch - 1}'
    if os.path.exists(f'{cur_models}\\discriminator.h5') and os.path.exists(f'{cur_models}\\generator.h5'):
        print('Building model from files...')
        d = load_model(f'{cur_models}\\discriminator.h5')
        g = load_model(f'{cur_models}\\generator.h5')

        if os.path.exists(f'{cur_models}\\d_g.h5'):
            d_on_g = load_model(f'{cur_models}\\d_g.h5')
        else:
            g_optim = Adam(lr=0.0001, beta_1=0.5)
            d_on_g = generator_containing_discriminator(g, d)
            d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    elif os.path.exists(f'{cur_models}\\discriminator.weights') and os.path.exists(
            f'{cur_models}\\generator.weights'):
        print('Building model with weights...')
        d_optim = Adam(lr=0.00001)
        d = build_discriminator(size)
        d.load_weights(f'{cur_models}\\discriminator.weights')
        d.compile(loss='binary_crossentropy', optimizer=d_optim, metrics=['accuracy'])

        g = build_generator(size)
        g.load_weights(f'{cur_models}\\generator.weights')

        g_optim = Adam(lr=0.0001, beta_1=0.5)
        d_on_g = generator_containing_discriminator(g, d)
        d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    else:
        print('Building model from scratch...')
        d_optim = Adam(lr=0.00001)
        g_optim = Adam(lr=0.0001, beta_1=0.5)

        d = build_discriminator(size)
        d.compile(loss='binary_crossentropy', optimizer=d_optim, metrics=['accuracy'])
        d.summary()

        g = build_generator(size)
        g.summary()

        d_on_g = generator_containing_discriminator(g, d)
        d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)

    if no_version:
        # Delete existing worlds and previews if any
        print('Checking for old generated data...')
        utils.delete_files_in_path(worlds_dir)
        utils.delete_files_in_path(previews_dir)

        print('Saving model images...')
        keras.utils.plot_model(d, to_file=f'{version_dir}\\discriminator.png', show_shapes=True,
                               show_layer_names=True)
        keras.utils.plot_model(g, to_file=f'{version_dir}\\generator.png', show_shapes=True, show_layer_names=True)

    # Load Data
    print('Loading worlds...')
    label_dict = utils.load_label_dict(res_dir, 'pro_labels_b')
    x_train = load_worlds_with_label(world_count, f'{res_dir}\\worlds\\', label_dict, 1, (size, size), block_forward)

    world_count = x_train.shape[0]
    batch_cnt = (world_count - (world_count % batch_size)) // batch_size

    # Set up tensorboard
    print('Setting up tensorboard...')
    tb_manager = TensorboardManager(graph_version_dir, batch_cnt)

    preview_frequency_sec = 5 * 60.0
    for epoch in range(initial_epoch, epochs):

        # Create directories for current epoch
        cur_worlds_dir = utils.check_or_create_local_path(f'epoch{epoch}', worlds_dir)
        cur_previews_dir = utils.check_or_create_local_path(f'epoch{epoch}', previews_dir)
        cur_models_dir = utils.check_or_create_local_path(f'epoch{epoch}', model_save_dir)

        print('Shuffling data...')
        np.random.shuffle(x_train)

        last_save_time = time.time()
        for batch in range(batch_cnt):

            # Get real set of images
            real_worlds = x_train[batch * batch_size:(batch + 1) * batch_size]

            # Get fake set of images
            noise = np.random.normal(0, 1, size=(batch_size, 256))
            fake_worlds = g.predict(noise)

            real_labels = np.ones((batch_size, 1))  # np.random.uniform(0.9, 1.1, size=(batch_size,))
            fake_labels = np.zeros((batch_size, 1))  # np.random.uniform(-0.1, 0.1, size=(batch_size,))

            # Train discriminator on real worlds
            d_loss = d.train_on_batch(real_worlds, real_labels)
            acc_real = d_loss[1]
            loss_real = d_loss[0]
            tb_manager.log_var('d_acc_real', epoch, batch, d_loss[1])
            tb_manager.log_var('d_loss_real', epoch, batch, d_loss[0])

            # Train discriminator on fake worlds
            d_loss = d.train_on_batch(fake_worlds, fake_labels)
            acc_fake = d_loss[1]
            loss_fake = d_loss[0]
            tb_manager.log_var('d_acc_fake', epoch, batch, d_loss[1])
            tb_manager.log_var('d_loss_fake', epoch, batch, d_loss[0])

            # Training generator on X data, with Y labels
            # noise = np.random.normal(0, 1, (batch_size, 256))

            # Train generator to generate real
            g_loss = d_on_g.train_on_batch(noise, real_labels)
            tb_manager.log_var('g_loss', epoch, batch, g_loss)

            print(f'epoch [{epoch}/{epochs}] :: batch [{batch}/{batch_cnt}] :: fake_acc = {acc_fake}:: '
                  f'real_acc = {acc_real} :: fake_loss = {loss_fake} :: real_loss = {loss_real} :: gen_loss = {g_loss}')

            # Save models
            time_since_save = time.time() - last_save_time
            if time_since_save >= preview_frequency_sec or batch == batch_cnt - 1:
                print('Saving previews...')
                for i in range(batch_size):
                    generated_world = fake_worlds[i]
                    decoded_world = utils.decode_world_sigmoid(block_backward, generated_world)
                    utils.save_world_data(decoded_world, f'{cur_worlds_dir}\\world{i}.world')
                    utils.save_world_preview(block_images, decoded_world, f'{cur_previews_dir}\\preview{i}.png')

                print('Saving models...')
                try:
                    d.save(f'{cur_models_dir}\\discriminator.h5')
                    g.save(f'{cur_models_dir}\\generator.h5')
                    d_on_g.save(f'{cur_models_dir}\\d_g.h5')
                    d.save_weights(f'{cur_models_dir}\\discriminator.weights')
                    g.save_weights(f'{cur_models_dir}\\generator.weights')
                    d_on_g.save_weights(f'{cur_models_dir}\\d_g.weights')
                except ImportError:
                    print('Failed to save data.')

                last_save_time = time.time()


def main():
    train(epochs=100, batch_size=64, world_count=16000, initial_epoch=0)


if __name__ == '__main__':
    main()
