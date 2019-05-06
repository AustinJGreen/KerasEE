import os

import keras
import numpy as np
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.models import Input
from keras.models import Model, Sequential, load_model

import utils
from loadworker import load_minimaps
from tbmanager import TensorboardManager


def build_basic_animator(size):
    # Takes in latent input, and target minimap colors
    # Outputs a real world whose minimap is supposed to reflect the target minimap

    animator_input = Input(shape=(size, size, 3))
    animator = Conv2D(filters=1024, kernel_size=(1, 1), strides=(1, 1), padding='same')(animator_input)
    animator = LeakyReLU()(animator)
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
        version_name = f'ver{latest + 1}'

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
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'blocks_optimized')

    print('Building model from scratch...')
    animator = build_basic_animator(sz)
    animator.compile(loss='mse', optimizer='adam')

    translator = load_model(f'{all_models_dir}\\translator\\ver15\\models\\best_loss.h5')
    translator.trainable = False

    animator_minimap = Sequential()
    animator_minimap.add(animator)
    animator_minimap.add(translator)
    animator_minimap.compile(loss='mse', optimizer='adam')

    print('Saving model images...')
    keras.utils.plot_model(animator, to_file=f'{version_dir}\\animator.png', show_shapes=True, show_layer_names=True)

    print('Loading worlds...')
    x_train = load_minimaps(world_count, f'{res_dir}\\worlds\\', (sz, sz), block_forward, mm_values)

    world_count = x_train.shape[0]
    batch_cnt = (world_count - (world_count % batch_size)) // batch_size

    # Set up tensorboard
    print('Setting up tensorboard...')
    tb_manager = TensorboardManager(graph_version_dir, batch_cnt)

    for epoch in range(epochs):

        # Create directories for current epoch
        cur_previews_dir = utils.check_or_create_local_path(f'epoch{epoch}', previews_dir)
        cur_models_dir = utils.check_or_create_local_path(f'epoch{epoch}', model_save_dir)

        print('Shuffling data...')
        np.random.shuffle(x_train)

        for batch in range(batch_cnt):
            minimaps = x_train[batch * batch_size:(batch + 1) * batch_size]
            # actual = y_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]

            # Train animator
            # world_loss = animator.train_on_batch(minimaps, actual)
            minimap_loss = animator_minimap.train_on_batch(minimaps, minimaps)
            tb_manager.log_var('mm_loss', epoch, batch, minimap_loss)

            print(f"Epoch = {epoch}/{epochs} :: Batch = {batch}/{batch_cnt} "
                  f":: MMLoss = {minimap_loss}")

            # Save previews and models
            if batch == batch_cnt - 1:
                print('Saving previews...')
                worlds = animator.predict(minimaps)
                trained = animator_minimap.predict(minimaps)
                for i in range(batch_size):
                    world_decoded = utils.decode_world_sigmoid(block_backward, worlds[i])
                    utils.save_world_preview(block_images, world_decoded, f'{cur_previews_dir}\\animated{i}.png')
                    utils.save_world_minimap(mm_values, world_decoded, f'{cur_previews_dir}\\actual{i}.png')
                    utils.save_rgb_map(utils.decode_world_minimap(trained[i]), f'{cur_previews_dir}\\trained{i}.png')

                    mm_decoded = utils.decode_world_minimap(minimaps[i])
                    utils.save_rgb_map(mm_decoded, f'{cur_previews_dir}\\target{i}.png')

                print('Saving models...')
                try:
                    animator.save(f'{cur_models_dir}\\animator.h5')
                    animator.save_weights(f'{cur_models_dir}\\animator.weights')
                except ImportError:
                    print('Failed to save data.')


def main():
    train(epochs=30, batch_size=1, world_count=10000, sz=112)
    # predict('ver9', dict_src_name='pro_labels')


if __name__ == '__main__':
    main()
