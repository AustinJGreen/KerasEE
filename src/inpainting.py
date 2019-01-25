import multiprocessing
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
    model_dir = utils.check_or_create_local_path("inpainting", all_models_dir)

    utils.delete_empty_versions(model_dir, 0)

    no_version = version_name is None
    if no_version:
        latest = utils.get_latest_version(model_dir)
        version_name = "ver%s" % (latest + 1)

    version_dir = utils.check_or_create_local_path(version_name, model_dir)
    graph_dir = utils.check_or_create_local_path("graph", model_dir)
    graph_version_dir = utils.check_or_create_local_path(version_name, graph_dir)

    worlds_dir = utils.check_or_create_local_path("worlds", version_dir)
    previews_dir = utils.check_or_create_local_path("previews", version_dir)
    model_save_dir = utils.check_or_create_local_path("models", version_dir)

    print("Saving source...")
    utils.save_source_to_dir(version_dir)

    # Load block images
    print("Loading block images...")
    block_images = utils.load_block_images(res_dir)

    print("Loading encoding dictionaries...")
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'optimized')

    # Load minimap values
    print("Loading minimap values...")
    minimap_values = utils.load_minimap_values(res_dir)

    # Load model
    print("Loading model...")
    feature_model = auto_encoder.autoencoder_model()
    feature_model.load_weights('%s\\auto_encoder\\ver12\\models\\epoch38\\autoencoder.weights' % all_models_dir)
    feature_layers = [7, 14, 21]

    contextnet = PConvUnet(feature_model, feature_layers, width=64, height=64, inference_only=False)
    unet = contextnet.build_pconv_unet(train_bn=True, lr=0.0001)
    unet.summary()
    # pconv_unet.load_weights('%s\\ver43\\models\\epoch4\\unet.weights' % contextnet_dir)

    if no_version:
        # Delete existing worlds and previews if any
        print("Checking for old generated data...")
        utils.delete_files_in_path(worlds_dir)
        utils.delete_files_in_path(previews_dir)

    print("Saving model images...")
    keras.utils.plot_model(unet, to_file="%s\\unet.png" % version_dir, show_shapes=True,
                           show_layer_names=True)

    # Set up tensorboard
    print("Setting up tensorboard...")
    tb_writer = tf.summary.FileWriter(logdir=graph_version_dir)
    unet_loss_summary = tf.Summary()
    unet_loss_summary.value.add(tag='unet_loss', simple_value=None)

    # Load Data
    cpu_count = multiprocessing.cpu_count()
    utilization_count = cpu_count - 1
    print("Loading worlds using %s cores." % utilization_count)
    x_train = load_worlds(world_count, "%s\\worlds\\" % res_dir, (64, 64), block_forward, utils.encode_world_sigmoid)

    # Start Training loop
    world_count = x_train.shape[0]
    number_of_batches = (world_count - (world_count % batch_size)) // batch_size

    for epoch in range(initial_epoch, epochs):

        print("Epoch = %s " % epoch)
        # Create directories for current epoch
        cur_worlds_cur = utils.check_or_create_local_path("epoch%s" % epoch, worlds_dir)
        cur_previews_dir = utils.check_or_create_local_path("epoch%s" % epoch, previews_dir)
        cur_models_dir = utils.check_or_create_local_path("epoch%s" % epoch, model_save_dir)

        print("Shuffling data...")
        np.random.shuffle(x_train)

        for minibatch_index in range(number_of_batches):

            # Get real set of images
            world_batch = x_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
            world_batch_masked, world_masks = utils.mask_batch_high(world_batch)

            if minibatch_index % 1000 == 999 or minibatch_index == number_of_batches - 1:

                # Save model
                try:
                    unet.save("%s\\unet.h5" % cur_models_dir)
                    unet.save_weights("%s\\unet.weights" % cur_models_dir)
                except ImportError:
                    print("Failed to save data.")

                # Save previews
                test = unet.predict([world_batch_masked, world_masks])

                d0 = utils.decode_world_sigmoid(block_backward, world_batch[0])
                utils.save_world_preview(block_images, d0, '%s\\%s_orig.png' % (cur_previews_dir, minibatch_index))

                d1 = utils.decode_world_sigmoid(block_backward, test[0])
                utils.save_world_preview(block_images, d1, '%s\\%s_fixed.png' % (cur_previews_dir, minibatch_index))

                d2 = utils.decode_world_sigmoid(block_backward, world_batch_masked[0])
                utils.save_world_preview(block_images, d2, '%s\\%s_masked.png' % (cur_previews_dir, minibatch_index))

            loss = unet.train_on_batch([world_batch_masked, world_masks], world_batch)

            unet_loss_summary.value[0].simple_value = loss / 1000.0  # Divide by 1000 for better Y-Axis values
            tb_writer.add_summary(unet_loss_summary, (epoch * number_of_batches) + minibatch_index)
            tb_writer.flush()

            print("epoch [%d/%d] :: batch [%d/%d] :: unet_loss = %f" % (
                epoch, epochs, minibatch_index, number_of_batches, loss))


def main():
    train(epochs=100, batch_size=1, world_count=60000, initial_epoch=0)


if __name__ == "__main__":
    main()
