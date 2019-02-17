import os

import keras
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam

import utils
from loadworker import load_worlds, load_world
from tbmanager import TensorboardManager


def autoencoder_model(size):
    model = Sequential(name='autoencoder')

    f = 64
    s = size

    while s > 7:
        if s == size:
            model.add(Conv2D(f, kernel_size=5, strides=1, padding='same', input_shape=(size, size, 10)))
        else:
            model.add(Conv2D(f, kernel_size=5, strides=1, padding='same'))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Conv2D(f, kernel_size=5, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        f = f * 2
        s = s // 2

    while s < size:
        model.add(Conv2DTranspose(f, kernel_size=5, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(f, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        f = f // 2
        s = s * 2

    model.add(Conv2DTranspose(10, kernel_size=5, strides=1, padding='same'))
    model.add(Activation('sigmoid'))

    model.trainable = True
    model.summary()
    return model


def train(epochs, batch_size, world_count, version_name=None):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    all_models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path('auto_encoder', all_models_dir)
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

    latest_epoch = utils.get_latest_epoch(model_save_dir)
    initial_epoch = latest_epoch + 1

    print('Saving source...')
    utils.save_source_to_dir(version_dir)

    # Load block images
    print('Loading block images...')
    block_images = utils.load_block_images(res_dir)

    print('Loading encoding dictionaries...')
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'blocks_optimized')

    # Load model and existing weights
    print('Loading model...')

    # Try to load full model, otherwise try to load weights
    loaded_model = False
    if not no_version and latest_epoch != -1:
        if os.path.exists(f'{version_dir}\\models\\epoch{latest_epoch}\\autoencoder.h5'):
            print('Found models.')
            ae = load_model(f'{version_dir}\\models\\epoch{latest_epoch}\\autoencoder.h5')
            loaded_model = True
        elif os.path.exists(f'{version_dir}\\models\\epoch{latest_epoch}\\autoencoder.weights'):
            print('Found weights.')
            ae = autoencoder_model(112)
            ae.load_weights(f'{version_dir}\\models\\epoch{latest_epoch}\\autoencoder.weights')

            print('Compiling model...')
            ae_optim = Adam(lr=0.0001)
            ae.compile(loss='binary_crossentropy', optimizer=ae_optim)
            loaded_model = True

    # Model was not loaded, compile new one
    if not loaded_model:
        print('Compiling model...')
        ae = autoencoder_model(112)
        print('Compiling model...')
        ae_optim = Adam(lr=0.0001)
        ae.compile(loss='binary_crossentropy', optimizer=ae_optim)

    if no_version:
        # Delete existing worlds and previews if any
        print('Checking for old generated data...')
        utils.delete_files_in_path(worlds_dir)
        utils.delete_files_in_path(previews_dir)

    print('Saving model images...')
    keras.utils.plot_model(ae, to_file=f'{version_dir}\\autoencoder.png', show_shapes=True, show_layer_names=True)

    # Load Data
    print('Loading worlds...')
    x_train = load_worlds(world_count, f'{res_dir}\\worlds\\', (112, 112), block_forward)

    # Start Training loop
    world_count = x_train.shape[0]
    batch_cnt = (world_count - (world_count % batch_size)) // batch_size

    # Set up tensorboard
    print('Setting up tensorboard...')
    tb_manager = TensorboardManager(graph_version_dir, batch_cnt)

    for epoch in range(initial_epoch, epochs):

        # Create directories for current epoch
        cur_worlds_cur = utils.check_or_create_local_path(f'epoch{epoch}', worlds_dir)
        cur_previews_dir = utils.check_or_create_local_path(f'epoch{epoch}', previews_dir)
        cur_models_dir = utils.check_or_create_local_path(f'epoch{epoch}', model_save_dir)

        print('Shuffling data...')
        np.random.shuffle(x_train)

        for batch in range(batch_cnt):

            # Get real set of images
            world_batch = x_train[batch * batch_size:(batch + 1) * batch_size]

            # Train
            loss = ae.train_on_batch(world_batch, world_batch)

            # Save snapshot of generated images on last batch
            if batch == batch_cnt - 1:

                # Generate samples
                generated = ae.predict(world_batch)

                # Save samples
                for image_num in range(batch_size):
                    generated_world = generated[image_num]
                    decoded_world = utils.decode_world_sigmoid(block_backward, generated_world)
                    utils.save_world_data(decoded_world, f'{cur_worlds_cur}\\world{image_num}.world')
                    utils.save_world_preview(block_images, decoded_world,
                                             f'{cur_previews_dir}\\preview{image_num}.png')

                # Save actual worlds
                for image_num in range(batch_size):
                    actual_world = world_batch[image_num]
                    decoded_world = utils.decode_world_sigmoid(block_backward, actual_world)
                    utils.save_world_preview(block_images, decoded_world,
                                             f'{cur_previews_dir}\\actual{image_num}.png')

            # Write loss
            tb_manager.log_var('ae_loss', epoch, batch, loss)

            print(f'epoch [{epoch}/{epochs}] :: batch [{batch}/{batch_cnt}] :: loss = {loss}')

            # Save models
            if batch % 100 == 99 or batch == batch_cnt - 1:
                print('Saving models...')
                try:
                    ae.save(f'{cur_models_dir}\\autoencoder.h5')
                    ae.save_weights(f'{cur_models_dir}\\autoencoder.weights')
                except ImportError:
                    print('Failed to save data.')


def predict_sample_matlab(network_ver, samples):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    all_models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path('auto_encoder', all_models_dir)
    version_dir = utils.check_or_create_local_path(network_ver, model_dir)
    model_save_dir = utils.check_or_create_local_path('models', version_dir)

    plots_dir = utils.check_or_create_local_path('plots', model_dir)
    utils.delete_files_in_path(plots_dir)

    print('Loading model...')
    latest_epoch = utils.get_latest_epoch(model_save_dir)
    auto_encoder = load_model(f'{model_save_dir}\\epoch{latest_epoch}\\autoencoder.h5')

    print('Loading block images...')
    block_images = utils.load_block_images(res_dir)

    print('Loading encoding dictionaries...')
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'blocks_optimized')

    x_worlds = os.listdir(f'{res_dir}\\worlds\\')
    np.random.shuffle(x_worlds)

    world_size = auto_encoder.input_shape[1]
    dpi = 96
    rows = samples
    cols = 2
    hpixels = 520 * cols
    hfigsize = hpixels / dpi
    vpixels = 530 * rows
    vfigsize = vpixels / dpi
    fig = plt.figure(figsize=(hfigsize, vfigsize), dpi=dpi)

    def set_ticks():
        no_labels = 2  # how many labels to see on axis x
        step = (16 * world_size) / (no_labels - 1)  # step between consecutive labels
        positions = np.arange(0, (16 * world_size) + 1, step)  # pixel count at label position
        labels = positions // 16
        plt.xticks(positions, labels)
        plt.yticks(positions, labels)

    sample_num = 0
    for world_filename in x_worlds:
        world_file = os.path.join(f'{res_dir}\\worlds\\', world_filename)
        world_id = utils.get_world_id(world_filename)

        # Load world and save preview
        encoded_regions = load_world(world_file, (world_size, world_size), block_forward)
        if len(encoded_regions) == 0:
            continue

        # Create prediction
        batch_input = np.empty((1, world_size, world_size, 10), dtype=np.int8)
        batch_input[0] = encoded_regions[0]
        encoded_world = auto_encoder.predict(batch_input)

        before = utils.decode_world_sigmoid(block_backward, encoded_regions[0])
        utils.save_world_preview(block_images, before, f'{plots_dir}\\before{sample_num}.png')

        after = utils.decode_world_sigmoid(block_backward, encoded_world[0])
        utils.save_world_preview(block_images, after, f'{plots_dir}\\after{sample_num}.png')

        # Create before plot
        before_img = mpimg.imread(f'{plots_dir}\\before{sample_num}.png')
        encoded_subplt = fig.add_subplot(rows, cols, sample_num + 1)
        encoded_subplt.set_title(f'{world_id}\nActual')
        set_ticks()
        plt.imshow(before_img)

        # Create after plot
        after_img = mpimg.imread(f'{plots_dir}\\after{sample_num}.png')
        encoded_subplt = fig.add_subplot(rows, cols, sample_num + 2)
        encoded_subplt.set_title(f'{world_id}\nEncoded')
        set_ticks()
        plt.imshow(after_img)

        print(f'Added plot {(sample_num / 2) + 1} of {samples}')

        sample_num += 2
        if sample_num >= rows * cols:
            break

    print('Saving figure...')
    fig.tight_layout()
    fig.savefig(f'{plots_dir}\\plot.png', transparent=True)


def main():
    # train(epochs=100, batch_size=32, world_count=30000, version_name='ver17')
    predict_sample_matlab('ver17', samples=4)


if __name__ == '__main__':
    main()
