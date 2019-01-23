import multiprocessing
import os

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.engine import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation, Dense, Reshape, Permute, Flatten, Lambda
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam

import utils
from loadworker import load_worlds


def build_encoder_layers(encoder_input):
    encoding = Conv2D(32, kernel_size=5, strides=1, padding='same')(encoder_input)
    encoding = BatchNormalization(momentum=0.8)(encoding)
    encoding = Activation('relu')(encoding)

    encoding = Conv2D(32, kernel_size=5, strides=1, padding='same')(encoding)
    encoding = BatchNormalization(momentum=0.8)(encoding)
    encoding = Activation('relu')(encoding)

    encoding = MaxPooling2D(pool_size=(2, 2))(encoding)  # 32x32x32

    encoding = Conv2D(64, kernel_size=5, strides=1, padding='same')(encoding)
    encoding = BatchNormalization(momentum=0.8)(encoding)
    encoding = Activation('relu')(encoding)

    encoding = Conv2D(64, kernel_size=5, strides=1, padding='same')(encoding)
    encoding = BatchNormalization(momentum=0.8)(encoding)
    encoding = Activation('relu')(encoding)

    encoding = MaxPooling2D(pool_size=(2, 2))(encoding)  # 16x16x64

    encoding = Conv2D(128, kernel_size=3, strides=1, padding='same')(encoding)
    encoding = BatchNormalization(momentum=0.8)(encoding)
    encoding = Activation('relu')(encoding)

    encoding = Conv2D(128, kernel_size=3, strides=1, padding='same')(encoding)
    encoding = BatchNormalization(momentum=0.8)(encoding)
    encoding = Activation('relu')(encoding)

    encoding = MaxPooling2D(pool_size=(2, 2))(encoding)  # 8x8x128

    encoding = Conv2D(256, kernel_size=3, strides=1, padding='same')(encoding)
    encoding = BatchNormalization(momentum=0.8)(encoding)
    encoding = Activation('relu')(encoding)

    encoding = Conv2D(256, kernel_size=3, strides=1, padding='same')(encoding)
    encoding = BatchNormalization(momentum=0.8)(encoding)
    encoding = Activation('relu')(encoding)

    encoding = MaxPooling2D(pool_size=(2, 2))(encoding)  # 4x4x256

    encoding = Flatten()(encoding)
    return encoding


def build_helper_model():
    view_input = Input(shape=(64, 64, 10))
    view_copy = Lambda(lambda x: x)(view_input)

    # First input takes in world and encodes world
    view = build_encoder_layers(view_input)

    latent_input = Input(shape=(128,))
    latent = Dense(units=4 * 4 * 256, activation='relu')(latent_input)

    decoder = Concatenate(axis=-1)([view, latent])  # Concatenate encoded neurons and random normal distribution
    decoder = Dense(units=4 * 4 * 256, activation='relu')(decoder)  # Dense connections between the two
    decoder = Reshape(target_shape=(4, 4, 256))(decoder)  # Reshape for decoding

    decoder = Conv2DTranspose(256, kernel_size=3, strides=1, padding='same')(decoder)
    decoder = BatchNormalization(momentum=0.8)(decoder)
    decoder = Activation('relu')(decoder)

    decoder = Conv2DTranspose(256, kernel_size=5, strides=2, padding='same')(decoder)  # 8x8
    decoder = BatchNormalization(momentum=0.8)(decoder)
    decoder = Activation('relu')(decoder)

    decoder = Conv2DTranspose(128, kernel_size=3, strides=1, padding='same')(decoder)
    decoder = BatchNormalization(momentum=0.8)(decoder)
    decoder = Activation('relu')(decoder)

    decoder = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(decoder)  # 16x16
    decoder = BatchNormalization(momentum=0.8)(decoder)
    decoder = Activation('relu')(decoder)

    decoder = Conv2DTranspose(64, kernel_size=3, strides=1, padding='same')(decoder)
    decoder = BatchNormalization(momentum=0.8)(decoder)
    decoder = Activation('relu')(decoder)

    decoder = Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(decoder)  # 32x32
    decoder = BatchNormalization(momentum=0.8)(decoder)
    decoder = Activation('relu')(decoder)

    decoder = Conv2DTranspose(32, kernel_size=3, strides=1, padding='same')(decoder)
    decoder = BatchNormalization(momentum=0.8)(decoder)
    decoder = Activation('relu')(decoder)

    decoder = Conv2DTranspose(32, kernel_size=5, strides=2, padding='same')(decoder)  # 64x64
    decoder = BatchNormalization(momentum=0.8)(decoder)
    decoder = Activation('relu')(decoder)

    decoder = Conv2DTranspose(10, kernel_size=3, strides=1, padding='same')(decoder)  # 64x64x10
    decoder = BatchNormalization(momentum=0.8)(decoder)
    generated = Activation('sigmoid')(decoder)

    helper_model = Model(name='helper', inputs=[view_input, latent_input], outputs=[view_copy, generated])
    keras.utils.plot_model(helper_model, to_file='C:\\Users\\austi\\Desktop\\helper.png', show_shapes=True,
                           show_layer_names=True)
    helper_model.summary()
    return helper_model


def build_judge_model():
    orig_input = Input(shape=(64, 64, 10))
    orig_encoding = Conv2D(256, kernel_size=5, strides=1, )(orig_input)
    orig_encoding = build_encoder_layers(orig_encoding)
    orig_latent = Activation(activation='sigmoid')(orig_encoding)

    generated_input = Input(shape=(64, 64, 10))
    generated_encoding = Conv2D(256, kernel_size=5, strides=1)(generated_input)
    generated_encoding = build_encoder_layers(generated_encoding)
    generated_latent = Activation(activation='sigmoid')(generated_encoding)

    both = Lambda(lambda x: K.abs(x[0] - x[1]))([orig_latent, generated_latent])
    prediction = Dense(64 * 64)(both)
    prediction = Reshape((1, 64 * 64))(prediction)
    prediction = Permute((2, 1))(prediction)
    prediction = Activation('sigmoid')(prediction)

    judge_model = Model(name='judge', inputs=[orig_input, generated_input], outputs=[prediction])
    judge_model.summary()
    return judge_model


def build_helper_feedback_model(helper, judge):
    judge.trainable = False

    view_input = Input(shape=(64, 64, 10))
    latent_input = Input(shape=(128,))

    helper_feedback = helper([view_input, latent_input])
    helper_feedback = judge(helper_feedback)

    helper_feedback_model = Model(inputs=[view_input, latent_input], outputs=helper_feedback)

    return helper_feedback_model


def train(epochs, batch_size, world_count, version_name=None, initial_epoch=0):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    all_models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path("helper", all_models_dir)

    utils.delete_empty_versions(model_dir, 1)
    if version_name is None:
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

    # Delete existing worlds and previews if any
    print("Checking for old generated data...")
    utils.delete_files_in_path(worlds_dir)
    utils.delete_files_in_path(previews_dir)

    # Load model and existing weights
    print("Loading models...")

    judge = build_judge_model()
    judge_optimizer = Adam(lr=0.00001)
    judge.compile(loss="binary_crossentropy", optimizer=judge_optimizer, metrics=['accuracy'])

    helper_optimizer = Adam(lr=0.0001)
    helper = build_helper_model()
    helper_feedback = build_helper_feedback_model(helper, judge)
    helper_feedback.compile(loss="binary_crossentropy", optimizer=helper_optimizer)

    # before training init writer (for tensorboard log) / model
    tb_writer = tf.summary.FileWriter(logdir=graph_version_dir)
    j_loss_summary = tf.Summary()
    j_loss_summary.value.add(tag='j_loss', simple_value=None)

    j_acc_summary = tf.Summary()
    j_acc_summary.value.add(tag='j_acc', simple_value=None)

    j_real_acc_summary = tf.Summary()
    j_real_acc_summary.value.add(tag='j_real_acc', simple_value=None)

    j_fake_acc_summary = tf.Summary()
    j_fake_acc_summary.value.add(tag='j_fake_acc', simple_value=None)

    h_loss_summary = tf.Summary()
    h_loss_summary.value.add(tag='h_loss', simple_value=None)

    # Load Data
    cpu_count = multiprocessing.cpu_count()
    utilization_count = cpu_count - 1
    print("Loading worlds using %s cores." % utilization_count)
    x_train = load_worlds(world_count, "%s\\world_repo\\" % res_dir, 64, 64, minimap_values, block_forward,
                          utilization_count)

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
            # Get real set of worlds
            world_batch = x_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
            world_batch_masked, world_masks = utils.mask_batch_low(world_batch)
            world_masks_reshaped = np.reshape(world_masks[:, :, :, 0], (batch_size, 64 * 64, 1))

            # Get fake set of worlds
            noise = np.random.normal(0, 1, size=(batch_size, 128))
            generated = helper.predict([world_batch_masked, noise])

            std_dev_labels = np.clip(np.random.normal(1, 0.5, size=(batch_size, 64 * 64, 1)), 0, 1)
            real_labels = np.ones((batch_size, 64 * 64, 1))
            masked_labels = 1 - world_masks_reshaped

            judge.trainable = True
            j_real = judge.train_on_batch([world_batch_masked, world_batch], std_dev_labels)

            j_real_acc_summary.value[0].simple_value = j_real[1]
            tb_writer.add_summary(j_real_acc_summary, (epoch * number_of_batches) + minibatch_index)

            j_fake = judge.train_on_batch([world_batch_masked, generated[1]], masked_labels)

            j_fake_acc_summary.value[0].simple_value = j_fake[1]
            tb_writer.add_summary(j_fake_acc_summary, (epoch * number_of_batches) + minibatch_index)

            j_acc_summary.value[0].simple_value = (j_real[1] + j_fake[1]) / 2
            tb_writer.add_summary(j_acc_summary, (epoch * number_of_batches) + minibatch_index)

            j_loss_summary.value[0].simple_value = (j_real[0] + j_fake[0]) / 2
            tb_writer.add_summary(j_loss_summary, (epoch * number_of_batches) + minibatch_index)

            judge.trainable = False
            h_loss = helper_feedback.train_on_batch([world_batch_masked, noise], real_labels)

            h_loss_summary.value[0].simple_value = h_loss
            tb_writer.add_summary(h_loss_summary, (epoch * number_of_batches) + minibatch_index)
            tb_writer.flush()

            print(
                "epoch [%d/%d] :: batch [%d/%d] :: j_fake_loss = %f :: j_fake_acc = %f :: h_loss = %f" % (
                    epoch, epochs, minibatch_index, number_of_batches, j_fake[0], j_fake[1], h_loss))

            if minibatch_index % 1000 == 999 or minibatch_index == number_of_batches - 1:
                actual_world = world_batch_masked[0]
                a_decoded = utils.decode_world2d_binary(block_backward, actual_world)
                utils.save_world_preview(block_images, a_decoded,
                                         '%s\\actual%s.png' % (cur_previews_dir, minibatch_index))

                gen_world = generated[1][0]
                decoded = utils.decode_world2d_binary(block_backward, gen_world)
                utils.save_world_preview(block_images, decoded,
                                         '%s\\preview%s.png' % (cur_previews_dir, minibatch_index))

                # Save models
                try:
                    judge.save("%s\\judge.h5" % cur_models_dir)
                    helper.save("%s\\helper.h5" % cur_models_dir)
                    judge.save_weights("%s\\judge.weights" % cur_models_dir)
                    helper.save_weights("%s\\helper.weights" % cur_models_dir)
                except ImportError:
                    print("Failed to save data.")


def main():
    train(epochs=100, batch_size=1, world_count=50000, initial_epoch=0)


if __name__ == "__main__":
    main()