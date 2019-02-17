import os

import keras
import numpy as np
import tensorflow as tf

import auto_encoder
import utils
from loadworker import load_worlds
from unet_model import PConvUnet


def train(epochs, batch_size, world_count, version_name=None, initial_epoch=0):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    all_models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path('inpainting', all_models_dir)

    utils.delete_empty_versions(model_dir, 0)

    no_version = version_name is None
    if no_version:
        latest = utils.get_latest_version(model_dir)
        version_name = f'ver{latest}'

    version_dir = utils.check_or_create_local_path(version_name, model_dir)
    graph_dir = utils.check_or_create_local_path('graph', model_dir)
    graph_version_dir = utils.check_or_create_local_path(version_name, graph_dir)

    worlds_dir = utils.check_or_create_local_path('worlds', version_dir)
    previews_dir = utils.check_or_create_local_path('previews', version_dir)
    model_save_dir = utils.check_or_create_local_path('models', version_dir)

    print('Saving source...')
    utils.save_source_to_dir(version_dir)

    # Load block images
    print('Loading block images...')
    block_images = utils.load_block_images(res_dir)

    print('Loading encoding dictionaries...')
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'blocks_optimized')

    # Load model
    print('Loading model...')
    feature_model = auto_encoder.autoencoder_model()
    feature_model.load_weights(f'{all_models_dir}\\auto_encoder\\ver15\\models\\epoch28\\autoencoder.weights')
    feature_layers = [7, 14, 21]

    contextnet = PConvUnet(feature_model, feature_layers, inference_only=False)
    unet = contextnet.build_pconv_unet(train_bn=True, lr=0.0001)
    unet.summary()
    # pconv_unet.load_weights(f'{contextnet_dir}\\ver43\\models\\epoch4\\unet.weights')

    if no_version:
        # Delete existing worlds and previews if any
        print('Checking for old generated data...')
        utils.delete_files_in_path(worlds_dir)
        utils.delete_files_in_path(previews_dir)

    print('Saving model images...')
    keras.utils.plot_model(unet, to_file=f'{version_dir}\\unet.png', show_shapes=True,
                           show_layer_names=True)

    # Set up tensorboard
    print('Setting up tensorboard...')
    tb_writer = tf.summary.FileWriter(logdir=graph_version_dir)
    unet_loss_summary = tf.Summary()
    unet_loss_summary.value.add(tag='unet_loss', simple_value=None)

    # Load Data
    x_train = load_worlds(world_count, f'{res_dir}\\worlds\\', (128, 128), block_forward)

    # Start Training loop
    world_count = x_train.shape[0]
    batch_cnt = (world_count - (world_count % batch_size)) // batch_size

    for epoch in range(initial_epoch, epochs):

        print(f'Epoch = {epoch}')
        # Create directories for current epoch
        cur_worlds_cur = utils.check_or_create_local_path(f'epoch{epoch}', worlds_dir)
        cur_previews_dir = utils.check_or_create_local_path(f'epoch{epoch}', previews_dir)
        cur_models_dir = utils.check_or_create_local_path(f'epoch{epoch}', model_save_dir)

        print('Shuffling data...')
        np.random.shuffle(x_train)

        for batch in range(batch_cnt):

            # Get real set of images
            world_batch = x_train[batch * batch_size:(batch + 1) * batch_size]
            world_batch_masked, world_masks = utils.mask_batch_high(world_batch)

            if batch % 1000 == 999 or batch == batch_cnt - 1:

                # Save model
                try:
                    unet.save(f'{cur_models_dir}\\unet.h5')
                    unet.save_weights(f'{cur_models_dir}\\unet.weights')
                except ImportError:
                    print('Failed to save data.')

                # Save previews
                test = unet.predict([world_batch_masked, world_masks])

                d0 = utils.decode_world_sigmoid(block_backward, world_batch[0])
                utils.save_world_preview(block_images, d0, f'{cur_previews_dir}\\{batch}_orig.png')

                d1 = utils.decode_world_sigmoid(block_backward, test[0])
                utils.save_world_preview(block_images, d1, f'{cur_previews_dir}\\{batch}_fixed.png')

                d2 = utils.decode_world_sigmoid(block_backward, world_batch_masked[0])
                utils.save_world_preview(block_images, d2, f'{cur_previews_dir}\\{batch}_masked.png')

            loss = unet.train_on_batch([world_batch_masked, world_masks], world_batch)

            unet_loss_summary.value[0].simple_value = loss / 1000.0  # Divide by 1000 for better Y-Axis values
            tb_writer.add_summary(unet_loss_summary, (epoch * batch_cnt) + batch)
            tb_writer.flush()

            print(f'epoch [{epoch}/{epochs}] :: batch [{batch}/{batch_cnt}] :: unet_loss = {loss}')


def main():
    train(epochs=100, batch_size=1, world_count=20000, initial_epoch=0)


if __name__ == '__main__':
    main()
