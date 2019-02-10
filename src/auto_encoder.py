import math
import os

import keras
import numpy as np
import tensorflow as tf
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam

import utils
from loadworker import load_worlds


def autoencoder_model(size):
    model = Sequential(name="autoencoder")

    f = 64
    s = size

    while s > 7:
        model.add(Conv2D(f, kernel_size=5, strides=1, padding="same", input_shape=(size, size, 10)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Conv2D(f, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        f = f * 2
        s = s // 2

    while s < size:
        model.add(Conv2DTranspose(f, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(f, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        f = f // 2
        s = s * 2

    model.add(Conv2DTranspose(10, kernel_size=5, strides=1, padding="same"))
    model.add(Activation('sigmoid'))

    model.trainable = True
    model.summary()
    return model


def train(epochs, batch_size, world_count, version_name=None):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    all_models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path("auto_encoder", all_models_dir)
    utils.delete_empty_versions(model_dir, 1)

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
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'blocks_optimized')

    # Load model and existing weights
    print("Loading model...")

    # Try to load full model, otherwise try to load weights
    if os.path.exists("%s\\autoencoder.h5" % version_dir):
        print("Found models.")
        ae = load_model("%s\\autoencoder.h5" % version_dir)
    elif os.path.exists("%s\\autoencoder.model" % version_dir):
        print("Found weights.")
        ae = autoencoder_model(112)
        ae.load_weights("%s\\autoencoder.model" % version_dir)

        print("Compiling model...")
        ae_optim = Adam(lr=0.0001)
        ae.compile(loss="binary_crossentropy", optimizer=ae_optim)
    else:
        print("Compiling model...")
        ae = autoencoder_model(112)
        print("Compiling model...")
        ae_optim = Adam(lr=0.0001)
        ae.compile(loss="binary_crossentropy", optimizer=ae_optim)

    if no_version:
        # Delete existing worlds and previews if any
        print("Checking for old generated data...")
        utils.delete_files_in_path(worlds_dir)
        utils.delete_files_in_path(previews_dir)

    print("Saving model images...")
    keras.utils.plot_model(ae, to_file="%s\\autoencoder.png" % version_dir, show_shapes=True, show_layer_names=True)

    # Set up tensorboard
    print("Setting up tensorboard...")
    tb_callback = keras.callbacks.TensorBoard(log_dir=graph_version_dir, write_graph=True)
    tb_callback.set_model(ae)

    # before training init writer (for tensorboard log) / model
    tb_writer = tf.summary.FileWriter(logdir=graph_version_dir)
    ae_loss = tf.Summary()
    ae_loss.value.add(tag='ae_loss', simple_value=None)

    # Load Data
    print("Loading worlds...")
    x_train = load_worlds(world_count, "%s\\worlds\\" % res_dir, (128, 128), block_forward)

    # Start Training loop
    world_count = x_train.shape[0]
    number_of_batches = (world_count - (world_count % batch_size)) // batch_size

    for epoch in range(epochs):

        # Create directories for current epoch
        cur_worlds_cur = utils.check_or_create_local_path("epoch%s" % epoch, worlds_dir)
        cur_previews_dir = utils.check_or_create_local_path("epoch%s" % epoch, previews_dir)
        cur_models_dir = utils.check_or_create_local_path("epoch%s" % epoch, model_save_dir)

        print("Shuffling data...")
        np.random.shuffle(x_train)

        for minibatch_index in range(number_of_batches):

            # Get real set of images
            world_batch = x_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]

            loss = ae.train_on_batch(world_batch, world_batch)

            # Save snapshot of generated images on last batch
            if minibatch_index == number_of_batches - 1:

                # Generate samples
                generated = ae.predict(world_batch)

                # Save samples
                for batchImage in range(batch_size):
                    generated_world = generated[batchImage]
                    decoded_world = utils.decode_world_sigmoid(block_backward, generated_world)
                    utils.save_world_data(decoded_world, "%s\\world%s.world" % (cur_worlds_cur, batchImage))
                    utils.save_world_preview(block_images, decoded_world,
                                             "%s\\preview%s.png" % (cur_previews_dir, batchImage))

                # Save actual worlds
                for batchImage in range(batch_size):
                    actual_world = world_batch[batchImage]
                    decoded_world = utils.decode_world_sigmoid(block_backward, actual_world)
                    utils.save_world_preview(block_images, decoded_world,
                                             "%s\\actual%s.png" % (cur_previews_dir, batchImage))

            # Write loss
            if not math.isnan(loss):
                ae_loss.value[0].simple_value = loss
                tb_writer.add_summary(ae_loss, (epoch * number_of_batches) + minibatch_index)

            print("epoch [%d/%d] :: batch [%d/%d] :: loss = %f" % (
                epoch, epochs, minibatch_index, number_of_batches, loss))

            # Save models
            if minibatch_index % 100 == 99 or minibatch_index == number_of_batches - 1:
                try:
                    ae.save("%s\\autoencoder.h5" % cur_models_dir)
                    ae.save_weights("%s\\autoencoder.weights" % cur_models_dir)
                except ImportError:
                    print("Failed to save data.")


def main():
    train(epochs=100, batch_size=32, world_count=50000)


if __name__ == "__main__":
    main()
