import os
import time

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Reshape
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
    return model


def build_discriminator():
    model = Sequential(name="discriminator")

    model.add(Conv2D(64, kernel_size=5, strides=1, padding="same"))
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

    model.add(Conv2D(512, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(Conv2D(512, kernel_size=5, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))

    model.trainable = True
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def improved_loss(generator, discriminator):

    def l1(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.sum(K.abs(y_pred - y_true), axis=[1, 2, 3])
        elif K.ndim(y_true) == 3:
            return K.sum(K.abs(y_pred - y_true), axis=[1, 2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")

    def gram_matrix(x):
        """Calculate gram matrix used in style loss"""

        # Assertions on input
        assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
        assert K.image_data_format() == 'channels_last', "Please use channels-last format"

        # Permute channels and get resulting shape
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        b, c, h, w = shape[0], shape[1], shape[2], shape[3]

        # Reshape x and do batch dot product
        features = K.reshape(x, K.stack([b, c, h * w]))
        gram = K.batch_dot(features, features, axes=2)

        # Normalize with channels, height and width
        gram = gram / K.cast(c * h * w, x.dtype)

        return gram

    def loss_style(arg1, arg2):
        """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
        style_loss = 0
        for o, g in zip(arg1, arg2):
            style_loss += l1(gram_matrix(o), gram_matrix(g))
        return style_loss

    def loss(y_true, y_pred):

        binary_crossentropy = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        d_true_features = d_feature_model(y_true)
        g_pred_features = g_feature_model(y_pred)

        feature_loss = loss_style(d_true_features, g_pred_features)

        return feature_loss + binary_crossentropy

    return loss


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
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'blocks_optimized')

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
            d_on_g.compile(loss=improved_loss(g, d), optimizer=g_optim)
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
        d_on_g.compile(loss=improved_loss(g, d), optimizer=g_optim)
    else:
        print("Building model from scratch...")
        d_optim = Adam(lr=0.00001, beta_1=0.5)
        g_optim = Adam(lr=0.0001, beta_1=0.5)

        d = build_discriminator()
        d.compile(loss="binary_crossentropy", optimizer=d_optim, metrics=["accuracy"])

        g = build_generator()
        d_on_g = generator_containing_discriminator(g, d)

        d_on_g.compile(loss=improved_loss(g, d), optimizer=g_optim)

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
    x_train = load_worlds(world_count, "%s\\worlds\\" % res_dir, (64, 64), block_forward)

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

            print("epoch [%d/%d] :: batch [%d/%d] :: dis_acc = %.1f%% :: dis_loss = %f :: gen_loss = %f" % (
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
    train(epochs=100, batch_size=100, world_count=100000, initial_epoch=0)


if __name__ == "__main__":
    main()
