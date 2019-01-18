import multiprocessing
import os
import random

import keras
import numpy as np
import tensorflow as tf
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam

from src import utils
from src.loadworker import GanWorldLoader


def load_worlds(load_count, world_directory, gen_width, gen_height, minimap_values, block_forward, thread_count):
    world_names = os.listdir(world_directory)
    random.shuffle(world_names)

    with multiprocessing.Manager() as manager:
        file_queue = manager.Queue()

        for name in world_names:
            file_queue.put(world_directory + name)

        world_array = np.zeros((load_count, gen_width, gen_height, 10), dtype=np.int8)

        world_counter = multiprocessing.Value('i', 0)
        thread_lock = multiprocessing.Lock()

        threads = [None] * thread_count
        for thread in range(thread_count):
            load_thread = GanWorldLoader(file_queue, manager, world_counter, thread_lock, load_count, gen_width,
                                         gen_height, block_forward, minimap_values)
            load_thread.start()
            threads[thread] = load_thread

        world_index = 0
        for thread in range(thread_count):
            threads[thread].join()
            print("Thread %s joined." % thread)
            thread_load_queue = threads[thread].get_worlds()
            print("Adding worlds to list from thread %s queue." % thread)
            while thread_load_queue.qsize() > 0:
                world_array[world_index] = thread_load_queue.get()
                world_index += 1
            print("Done adding worlds to list from thread.")

    return world_array


def autoencoder_model():
    model = Sequential(name="autoencoder")

    model.add(Conv2D(64, kernel_size=3, strides=1, padding="same", input_shape=(64, 64, 10)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2DTranspose(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(32, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(10, kernel_size=5, strides=1, padding="same"))
    model.add(Activation('sigmoid'))

    model.trainable = True
    model.summary()
    return model


def train(epochs, batch_size, world_count, version_name=None):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path("auto_encoder", models_dir)

    if version_name is None:
        latest = utils.get_latest_version(model_dir)
        version_name = "ver%s" % (latest + 1)

    version_dir = utils.check_or_create_local_path(version_name, model_dir)
    graph_dir = utils.check_or_create_local_path("graph", model_dir)
    graph_version_dir = utils.check_or_create_local_path(version_name, graph_dir)

    worlds_dir = utils.check_or_create_local_path("worlds", version_dir)
    previews_dir = utils.check_or_create_local_path("previews", version_dir)
    models_dir = utils.check_or_create_local_path("models", version_dir)

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

    # Load model and existing weights
    print("Loading model...")
    ae = None

    # Try to load full model, otherwise try to load weights
    if os.path.exists("%s\\autoencoder.h5" % version_dir):
        print("Found models.")
        ae = load_model("%s\\autoencoder.h5" % version_dir)
    elif os.path.exists("%s\\autoencoder.model" % version_dir):
        print("Found weights.")
        ae = autoencoder_model()
        ae.load_weights("%s\\autoencoder.model" % version_dir)

        print("Compiling model...")
        ae_optim = Adam(lr=0.0001)
        ae.compile(loss="binary_crossentropy", optimizer=ae_optim)
    else:
        print("Compiling model...")
        ae = autoencoder_model()
        print("Compiling model...")
        ae_optim = Adam(lr=0.0001)
        ae.compile(loss="binary_crossentropy", optimizer=ae_optim)

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
    cpu_count = multiprocessing.cpu_count()
    utilization_count = cpu_count - 1
    print("Loading worlds using %s cores." % utilization_count)
    x_train = load_worlds(world_count, "%s\\world_repo\\" % res_dir, 64, 64, minimap_values, block_forward,
                          utilization_count)

    # Start Training loop
    number_of_batches = world_count // batch_size

    for epoch in range(epochs):

        # Create directories for current epoch
        cur_worlds_cur = utils.check_or_create_local_path("epoch%s" % epoch, worlds_dir)
        cur_previews_dir = utils.check_or_create_local_path("epoch%s" % epoch, previews_dir)
        cur_models_dir = utils.check_or_create_local_path("epoch%s" % epoch, models_dir)

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
                    decoded_world = utils.decode_world2d_binary(block_backward, generated_world)
                    utils.save_world_data(decoded_world, "%s\\world%s.dat" % (cur_worlds_cur, batchImage))
                    utils.save_world_preview(block_images, decoded_world,
                                             "%s\\preview%s.png" % (cur_previews_dir, batchImage))

                # Save actual worlds
                for batchImage in range(batch_size):
                    actual_world = world_batch[batchImage]
                    decoded_world = utils.decode_world2d_binary(block_backward, actual_world)
                    utils.save_world_preview(block_images, decoded_world,
                                             "%s\\actual%s.png" % (cur_previews_dir, batchImage))

            # Write loss
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
    train(epochs=100, batch_size=64, world_count=100000)


if __name__ == "__main__":
    main()
