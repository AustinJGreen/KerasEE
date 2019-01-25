import os

import keras
import numpy as np
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential

import utils
import wgan_model
from loadworker import load_worlds

TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.


def build_generator():
    model = Sequential(name="generator")

    model.add(Dense(input_dim=100, units=4 * 4 * 512))
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
    model.add(Activation('tanh'))

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


def train(epochs, batch_size, world_count, version_name=None, initial_epoch=0):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    all_models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path("wgan", all_models_dir)

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

    print("Compiling model...")
    g = build_generator()
    d = build_discriminator()
    discriminator_model, generator_model, generator = wgan_model.build_wgan(batch_size, g, d, 100, (64, 64, 10))

    if no_version:
        # Delete existing worlds and previews if any
        print("Checking for old generated data...")
        utils.delete_files_in_path(worlds_dir)
        utils.delete_files_in_path(previews_dir)

        print("Saving model images...")
        keras.utils.plot_model(d, to_file="%s\\discriminator.png" % version_dir, show_shapes=True,
                               show_layer_names=True)
        keras.utils.plot_model(g, to_file="%s\\generator.png" % version_dir, show_shapes=True, show_layer_names=True)

    # Load Data
    print("Loading worlds...")
    x_train = load_worlds(world_count, "%s\\worlds\\" % res_dir, (64, 64), block_forward, utils.encode_world_tanh)

    # Start Training loop
    world_count = x_train.shape[0]
    number_of_batches = int(world_count // (batch_size * TRAINING_RATIO))
    minibatches_size = batch_size * TRAINING_RATIO

    # We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
    # negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
    # gradient_penalty loss function and is not used.
    positive_y = np.ones((batch_size, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

    for epoch in range(initial_epoch, epochs):

        print("Shuffling data...")
        np.random.shuffle(x_train)

        print("Epoch: ", epoch)
        for i in range(number_of_batches):
            discriminator_minibatches = x_train[i * minibatches_size:(i + 1) * minibatches_size]
            for j in range(TRAINING_RATIO):
                image_batch = discriminator_minibatches[j * batch_size:(j + 1) * batch_size]
                noise = np.random.rand(batch_size, 100).astype(np.float32)
                d_loss = discriminator_model.train_on_batch([image_batch, noise], [positive_y, negative_y, dummy_y])
                print("d_loss = %s" % d_loss)
            g_loss = generator_model.train_on_batch(np.random.rand(batch_size, 100), positive_y)
            print("g_loss = %s" % g_loss)

        print("Saving worlds...")
        fake_worlds = generator.predict(np.random.rand(5, 100))
        for batchImage in range(5):
            generated_world = fake_worlds[batchImage]
            decoded_world = utils.decode_world_sigmoid(block_backward, generated_world)
            utils.save_world_data(decoded_world, "%s\\world%s.world" % (worlds_dir, batchImage))
            utils.save_world_preview(block_images, decoded_world, "%s\\preview%s.png" % (previews_dir, batchImage))


def main():
    train(epochs=100, batch_size=64, world_count=1000)


if __name__ == "__main__":
    main()
