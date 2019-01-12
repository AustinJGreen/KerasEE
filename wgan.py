import multiprocessing
import os
import random

import numpy as np

import model
import utils
from loadworker import GanWorldLoader

TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.


def load_worlds(load_count, world_directory, gen_width, gen_height, block_forward_dict, minimap_values, thread_count):
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
                                         gen_height, block_forward_dict, minimap_values)
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


def train(epochs, batch_size, world_count, version_name=None):
    cur_dir = os.getcwd()
    gan_dir = utils.check_or_create_local_path("wgan")

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

    # Load preprocessing values
    print("Loading preprocessing values...")
    block_forward_dict, block_backward_dict = utils.load_encoding_dict("blocklist")

    print("Compiling model...")
    discriminator_model, generator_model, generator = model.build_wgan(batch_size)

    print("Loading data...")
    x_train = load_worlds(load_count=50000, world_directory='%s\\WorldRepo3\\' % cur_dir, gen_width=64, gen_height=64,
                          block_forward_dict=block_forward_dict,
                          minimap_values=minimap_values, thread_count=1)

    # for i in range(x_train.shape[0]):
    #    encoded = x_train[i]
    #    decoded = utils.decode_world2d_binary(encoded)
    #    utils.save_world_preview(block_images, decoded, 'C:\\Users\\austi\\Desktop\\previews\\%s.png' % i)

    # We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
    # negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
    # gradient_penalty loss function and is not used.
    positive_y = np.ones((batch_size, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

    for epoch in range(100):
        np.random.shuffle(x_train)
        print("Epoch: ", epoch)
        minibatches_size = batch_size * TRAINING_RATIO
        for i in range(int(x_train.shape[0] // (batch_size * TRAINING_RATIO))):
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
            decoded_world = utils.decode_world2d_binary(generated_world)
            utils.save_world_data(decoded_world, "%s\\world%s.dat" % (worlds_dir, batchImage))
            utils.save_world_preview(block_images, decoded_world, "%s\\preview%s.png" % (previews_dir, batchImage))


def main():
    train(epochs=1000, batch_size=64, world_count=100000)


if __name__ == "__main__":
    main()
