import os
import time

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam

import utils
from loadworker import load_worlds


def build_generator():
    model = Sequential(name="generator")

    model.add(Dense(input_dim=256, units=4 * 4 * 512))
    model.add(LeakyReLU())

    model.add(Reshape((4, 4, 512)))

    model.add(Conv2DTranspose(512, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(512, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(256, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(256, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(10, kernel_size=5, strides=1, padding="same"))
    model.add(Activation('sigmoid'))

    model.trainable = True
    model.summary()
    return model


def build_discriminator():
    model = Sequential(name="discriminator")

    model.add(Conv2D(64, kernel_size=5, strides=1, padding="same", input_shape=(64, 64, 10)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(Conv2D(64, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(Conv2D(128, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(Conv2D(256, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))

    model.trainable = True
    model.summary()
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
    model_dir = utils.check_or_create_local_path("gan", all_models_dir)

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

    print("Loading block images...")
    block_images = utils.load_block_images(res_dir)

    print("Loading encoding dictionaries...")
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'optimized')

    print("Loading minimap values...")
    minimap_values = utils.load_minimap_values(res_dir)

    # Load model and existing weights
    print("Loading model...")

    # Try to load full model, otherwise try to load weights
    cur_models = "%s\\epoch%s" % (model_save_dir, initial_epoch - 1)
    if os.path.exists("%s\\discriminator.h5" % cur_models) and os.path.exists("%s\\generator.h5" % cur_models):
        print("Building model from files...")
        d = load_model("%s\\discriminator.h5" % cur_models)
        g = load_model("%s\\generator.h5" % cur_models)

        if os.path.exists("%s\\d_g.h5" % cur_models):
            d_on_g = load_model("%s\\d_g.h5" % cur_models)
        else:
            g_optim = Adam(lr=0.0001, beta_1=0.5)
            d_on_g = generator_containing_discriminator(g, d)
            d_on_g.compile(loss="binary_crossentropy", optimizer=g_optim)
    elif os.path.exists("%s\\discriminator.weights" % cur_models) and os.path.exists(
            "%s\\generator.weights" % cur_models):
        print("Building model with weights...")
        d_optim = Adam(lr=0.00001)
        d = build_discriminator()
        d.load_weights("%s\\discriminator.weights" % cur_models)
        d.compile(loss="binary_crossentropy", optimizer=d_optim, metrics=["accuracy"])

        g = build_generator()
        g.load_weights("%s\\generator.weights" % cur_models)

        g_optim = Adam(lr=0.0001, beta_1=0.5)
        d_on_g = generator_containing_discriminator(g, d)
        d_on_g.compile(loss="binary_crossentropy", optimizer=g_optim)
    else:
        print("Building model from scratch...")
        d_optim = Adam(lr=0.00001)
        g_optim = Adam(lr=0.0001, beta_1=0.5)

        d = build_discriminator()
        d.compile(loss="binary_crossentropy", optimizer=d_optim, metrics=["accuracy"])

        g = build_generator()
        d_on_g = generator_containing_discriminator(g, d)

        d_on_g.compile(loss="binary_crossentropy", optimizer=g_optim)

    if no_version:
        # Delete existing worlds and previews if any
        print("Checking for old generated data...")
        utils.delete_files_in_path(worlds_dir)
        utils.delete_files_in_path(previews_dir)

        print("Saving model images...")
        keras.utils.plot_model(d, to_file="%s\\discriminator.png" % version_dir, show_shapes=True,
                               show_layer_names=True)
        keras.utils.plot_model(g, to_file="%s\\generator.png" % version_dir, show_shapes=True, show_layer_names=True)

    # Set up tensorboard
    print("Setting up tensorboard...")
    tb_callback = keras.callbacks.TensorBoard(log_dir=graph_version_dir, write_graph=True)
    tb_callback.set_model(d_on_g)

    # before training init writer (for tensorboard log) / model
    tb_writer = tf.summary.FileWriter(logdir=graph_version_dir)
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
    print("Loading worlds...")
    x_train = load_worlds(world_count, "%s\\worlds\\" % res_dir, (64, 64), block_forward, utils.encode_world_sigmoid)

    # Start Training loop
    world_count = x_train.shape[0]
    number_of_batches = (world_count - (world_count % batch_size)) // batch_size

    preview_frequency_sec = 5 * 60.0

    for epoch in range(initial_epoch, epochs):

        # Create directories for current epoch
        cur_worlds_cur = utils.check_or_create_local_path("epoch%s" % epoch, worlds_dir)
        cur_previews_dir = utils.check_or_create_local_path("epoch%s" % epoch, previews_dir)
        cur_models_dir = utils.check_or_create_local_path("epoch%s" % epoch, model_save_dir)

        print("Shuffling data...")
        np.random.shuffle(x_train)

        last_save_time = time.time()
        for minibatch_index in range(number_of_batches):

            # Get real set of images
            real_worlds = x_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]

            # Get fake set of images
            noise = np.random.normal(0, 1, size=(batch_size, 256))
            fake_worlds = g.predict(noise)

            real_labels = np.ones((batch_size, 1))  # np.random.uniform(0.9, 1.1, size=(batch_size,))
            fake_labels = np.zeros((batch_size, 1))  # np.random.uniform(-0.1, 0.1, size=(batch_size,))

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
            noise = np.random.normal(0, 1, (batch_size, 256))

            # Train generator to generate real
            g_loss = d_on_g.train_on_batch(noise, real_labels)
            g_loss_summary.value[0].simple_value = g_loss
            tb_writer.add_summary(g_loss_summary, (epoch * number_of_batches) + minibatch_index)
            tb_writer.flush()

            print("epoch [%d/%d] :: batch [%d/%d] :: disAcc = %.1f%% :: disLoss = %f :: genLoss = %f" % (
                epoch, epochs, minibatch_index, number_of_batches, d_avg_acc * 100, d_avg_loss, g_loss))

            # Save models
            time_since_save = time.time() - last_save_time
            if time_since_save >= preview_frequency_sec or minibatch_index == number_of_batches - 1:
                print("Saving previews...")
                for batchImage in range(batch_size):
                    generated_world = fake_worlds[batchImage]
                    decoded_world = utils.decode_world_sigmoid(block_backward, generated_world)
                    utils.save_world_data(decoded_world, "%s\\world%s.world" % (cur_worlds_cur, batchImage))
                    utils.save_world_preview(block_images, decoded_world,
                                             "%s\\preview%s.png" % (cur_previews_dir, batchImage))

                print("Saving models...")
                try:
                    d.save("%s\\discriminator.h5" % cur_models_dir)
                    g.save("%s\\generator.h5" % cur_models_dir)
                    d_on_g.save("%s\\d_g.h5" % cur_models_dir)
                    d.save_weights("%s\\discriminator.weights" % cur_models_dir)
                    g.save_weights("%s\\generator.weights" % cur_models_dir)
                    d_on_g.save_weights("%s\\d_g.weights" % cur_models_dir)
                except ImportError:
                    print("Failed to save data.")

                last_save_time = time.time()


def main():
    train(epochs=100, batch_size=50, world_count=200000, initial_epoch=0)


if __name__ == "__main__":
    main()
