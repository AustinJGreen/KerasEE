import multiprocessing
import os
import random

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.layers import LeakyReLU, Dropout
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, SGD

import utils
from loadworker import GanWorldLoader


# from keras.utils.generic_utils import get_custom_objects


def load_worlds(load_count, world_directory, gen_width, gen_height, minimap_values, thread_count):
    world_names = os.listdir(world_directory)
    random.shuffle(world_names)

    with multiprocessing.Manager() as manager:
        file_queue = manager.Queue()

        for name in world_names:
            file_queue.put(world_directory + name)

        world_array = np.zeros((load_count, gen_width, gen_height, 11), dtype=np.int8)

        world_counter = multiprocessing.Value('i', 0)
        thread_lock = multiprocessing.Lock()

        threads = [None] * thread_count
        for thread in range(thread_count):
            load_thread = GanWorldLoader(file_queue, manager, world_counter, thread_lock, load_count, gen_width,
                                         gen_height, None, minimap_values)
            load_thread.start()
            threads[thread] = load_thread

        world_index = 0
        for thread in range(thread_count):
            threads[thread].join()

            thread_load_queue = threads[thread].get_worlds()
            while thread_load_queue.qsize() > 0:
                world_array[world_index] = thread_load_queue.get()
                world_index += 1

    return world_array


def binary_activation(x):
    return keras.backend.greater(0.5, x)


def generator_model():
    model = Sequential(name="generator")
    model.add(Dense(input_dim=256, units=4 * 4 * 512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 512)))
    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(scale=False))
    # model.add(Dropout(0.25))
    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(scale=False))
    # model.add(Dropout(0.25))
    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(scale=False))
    # model.add(Dropout(0.25))
    model.add(Conv2DTranspose(11, kernel_size=5, strides=2, padding="same", kernel_initializer="glorot_normal"))
    model.add(Activation(binary_activation))
    model.trainable = True
    model.summary()
    return model


def discriminator_model():
    model = Sequential(name="discriminator")
    model.add(keras.layers.InputLayer(input_shape=(64, 64, 11)))
    model.add(Conv2D(16, kernel_size=5, strides=1, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=5, strides=1, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size=5, strides=1, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(128, kernel_size=5, strides=1, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(256, kernel_size=5, strides=1, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid", kernel_initializer="glorot_normal"))
    model.trainable = True
    model.summary()
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def train(epochs, batch_size, world_count, version_name=None):
    cur_dir = os.getcwd()
    gan_dir = utils.check_or_create_local_path("gan")

    if version_name is None:
        latest = utils.get_latest_version(gan_dir)
        version_name = "ver%s" % (latest + 1)

    version_dir = utils.check_or_create_local_path(version_name, gan_dir)
    graph_dir = utils.check_or_create_local_path("graph", version_dir)
    worlds_dir = utils.check_or_create_local_path("worlds", version_dir)
    previews_dir = utils.check_or_create_local_path("previews", version_dir)
    models_dir = utils.check_or_create_local_path("models", version_dir)

    print("Saving source...")
    utils.save_source_to_dir(version_dir)

    # Load block images
    print("Loading block images...")
    block_images = utils.load_block_images()

    # Load minimap values
    print("Loading minimap values...")
    minimap_values = utils.load_minimap_values()

    # Load model and existing weights
    print("Loading model...")
    d = None
    g = None
    d_on_g = None

    # Try to load full model, otherwise try to load weights
    '''if os.path.exists("%s\\discriminator.h5" % versionDir) and os.path.exists("%s\\generator.h5" % versionDir):
        print("Found models.")
        d = load_model("%s\\discriminator.h5" % versionDir)
        g = load_model("%s\\generator.h5" % versionDir)
        d_on_g = generator_containing_discriminator(g, d)
    else:
        #Load any existing weights if any
        if os.path.exists("%s\\discriminator.model" % versionDir) and os.path.exists("%s\\generator.model" % versionDir):
            print("Found weights.")
            d.load_weights("%s\\discriminator.model" % versionDir)
            g.load_weights("%s\\generator.model" % versionDir)'''

    print("Compiling model...")
    d_optim = SGD(lr=0.0001)
    g_optim = Adam(lr=0.0001, beta_1=0.5)

    d = discriminator_model()
    d.compile(loss="binary_crossentropy", optimizer=d_optim, metrics=["accuracy"])

    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)

    d_on_g.compile(loss="binary_crossentropy", optimizer=g_optim)

    # Delete existing worlds and previews if any
    print("Checking for old generated data...")
    utils.delete_files_in_path(worlds_dir)
    utils.delete_files_in_path(previews_dir)

    # print("Saving model images...")
    keras.utils.plot_model(d, to_file="%s\\discriminator.png" % version_dir, show_shapes=True, show_layer_names=True)
    keras.utils.plot_model(g, to_file="%s\\generator.png" % version_dir, show_shapes=True, show_layer_names=True)

    # Set up tensorboard
    print("Setting up tensorboard...")
    tb_callback = keras.callbacks.TensorBoard(log_dir=graph_dir, write_graph=True)
    tb_callback.set_model(d_on_g)

    # before training init writer (for tensorboard log) / model
    tb_writer = tf.summary.FileWriter(logdir=graph_dir)
    d_acc_summary = tf.Summary()
    d_acc_summary.value.add(tag='d_acc', simple_value=None)
    d_acc_real_summary = tf.Summary()
    d_acc_real_summary.value.add(tag='d_acc_real', simple_value=None)
    d_acc_fake_summary = tf.Summary()
    d_acc_fake_summary.value.add(tag='d_acc_fake', simple_value=None)
    d_loss_summary = tf.Summary()
    d_loss_summary.value.add(tag='d_loss', simple_value=None)
    d_loss_real_summary = tf.Summary()
    d_loss_real_summary.value.add(tag='d_loss_real', simple_value=None)
    d_loss_fake_summary = tf.Summary()
    d_loss_fake_summary.value.add(tag='d_loss_fake', simple_value=None)
    g_loss_summary = tf.Summary()
    g_loss_summary.value.add(tag='g_loss', simple_value=None)

    # Load Data
    cpu_count = multiprocessing.cpu_count()
    utilization_count = cpu_count - 1
    print("Loading worlds using %s cores." % utilization_count)
    x_train = load_worlds(world_count, "%s\\WorldRepo3\\" % cur_dir, 64, 64, minimap_values,
                          utilization_count)

    # Start Training loop
    number_of_batches = int(world_count / batch_size)

    # Initialize tables for Hashtable tensors
    K.get_session().run(tf.tables_initializer())

    for epoch in range(epochs):

        # Create directories for current epoch
        cur_worlds_cur = utils.check_or_create_local_path("epoch%s" % epoch, worlds_dir)
        cur_previews_dir = utils.check_or_create_local_path("epoch%s" % epoch, previews_dir)
        cur_models_dir = utils.check_or_create_local_path("epoch%s" % epoch, models_dir)

        print("Shuffling data...")
        np.random.shuffle(x_train)

        for minibatch_index in range(number_of_batches):

            # Get real set of images
            real_worlds = x_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]

            # Get fake set of images
            noise = np.random.normal(0, 0.5, size=(batch_size, 256))
            fake_worlds = g.predict(noise)

            real_labels = np.ones((batch_size, 1))  # np.random.uniform(0.9, 1.1, size=(batch_size,))
            fake_labels = np.zeros((batch_size, 1))  # np.random.uniform(-0.1, 0.1, size=(batch_size,))

            # Save snapshot of generated images on last batch
            if minibatch_index == number_of_batches - 1:
                for batchImage in range(batch_size):
                    generated_world = fake_worlds[batchImage]
                    decoded_world = utils.decode_world2d_binary(generated_world)
                    utils.save_world_data(decoded_world, "%s\\world%s.dat" % (cur_worlds_cur, batchImage))
                    utils.save_world_preview(block_images, decoded_world,
                                             "%s\\preview%s.png" % (cur_previews_dir, batchImage))

            d.trainable = True

            # Train discriminator on real worlds
            d_loss = d.train_on_batch(real_worlds, real_labels)
            d_real_acc = d_loss[1]
            d_real_loss = d_loss[0]

            # Write accuracy for real labels
            d_acc_real_summary.value[0].simple_value = d_real_acc
            tb_writer.add_summary(d_acc_real_summary, (epoch * number_of_batches) + minibatch_index)

            # Write loss for real labels
            d_loss_real_summary.value[0].simple_value = d_real_loss
            tb_writer.add_summary(d_loss_real_summary, (epoch * number_of_batches) + minibatch_index)

            # Train discriminator on fake worlds
            d_loss = d.train_on_batch(fake_worlds, fake_labels)
            d_fake_acc = d_loss[1]
            d_fake_loss = d_loss[0]

            # Write accuracy for fake labels
            d_acc_fake_summary.value[0].simple_value = d_fake_acc
            tb_writer.add_summary(d_acc_fake_summary, (epoch * number_of_batches) + minibatch_index)

            # Write loss for fake labels
            d_loss_fake_summary.value[0].simple_value = d_fake_loss
            tb_writer.add_summary(d_loss_fake_summary, (epoch * number_of_batches) + minibatch_index)

            # Calculate average loss and accuracy
            d_avg_acc = (d_real_acc + d_fake_acc) / 2.0
            d_avg_loss = (d_real_loss + d_fake_loss) / 2.0

            # Write average accuracy for real and fake labels
            d_acc_summary.value[0].simple_value = d_avg_acc
            tb_writer.add_summary(d_acc_summary, (epoch * number_of_batches) + minibatch_index)

            # Write average loss for real and fake labels
            d_loss_summary.value[0].simple_value = d_avg_loss
            tb_writer.add_summary(d_loss_summary, (epoch * number_of_batches) + minibatch_index)

            d.trainable = False

            # Training generator on X data, with Y labels
            noise = np.random.normal(0, 0.5, (batch_size, 256))

            # Train generator to generate real
            g_loss = d_on_g.train_on_batch(noise, real_labels)
            g_loss_summary.value[0].simple_value = g_loss
            tb_writer.add_summary(g_loss_summary, (epoch * number_of_batches) + minibatch_index)
            tb_writer.flush()

            print("epoch [%d/%d] :: batch [%d/%d] :: disAcc = %.1f%% :: disLoss = %f :: genLoss = %f" % (
                epoch, epochs, minibatch_index, number_of_batches, d_avg_acc * 100, d_avg_loss, g_loss))

            # Save models
            if minibatch_index % 100 == 99 or minibatch_index == number_of_batches - 1:
                try:
                    d.save("%s\\discriminator.h5" % cur_models_dir)
                    g.save("%s\\generator.h5" % cur_models_dir)
                    d.save_weights("%s\\discriminator.weights" % cur_models_dir)
                    g.save_weights("%s\\generator.weights" % cur_models_dir)
                except:
                    print("Failed to save data.")
                    pass

            # Debugging
            # if d_loss[1] == 1 or d_loss[0] == 0:
            #    batchesLasted = (epoch * numberOfBatches) + minibatchIndex + 1
            #    print ("Tune lasted %s" % batchesLasted)
            #    return

        # Perform some checks on the gradients of each model
        # check_grads(d, "discriminator")
        # check_grads(g, "generator")


def main():
    train(epochs=1000, batch_size=64, world_count=1000)


if __name__ == "__main__":
    main()
