import multiprocessing
import os
import random

import numpy as np

import ae
import unet
import utils
from loadworker import GanWorldLoader


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
            print("Thread %s joined." % thread)
            thread_load_queue = threads[thread].get_worlds()
            print("Adding worlds to list from thread %s queue." % thread)
            while thread_load_queue.qsize() > 0:
                world_array[world_index] = thread_load_queue.get()
                world_index += 1
            print("Done adding worlds to list from thread.")

    return world_array


def train(epochs, batch_size, world_count, version_name=None):
    cur_dir = os.getcwd()
    gan_dir = utils.check_or_create_local_path("contextnet")

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

    # Load model
    print("Loading model...")
    feature_model = ae.autoencoder_model()
    feature_model.load_weights('%s\\ae\\ver5\\models\\epoch30\\autoencoder.weights' % cur_dir)
    feature_layers = [7, 14, 21]

    contextnet = unet.PConvUnet(feature_model, feature_layers, width=64, height=64, inference_only=False)
    pconv_unet = contextnet.build_pconv_unet(train_bn=True, lr=0.0001)

    # Delete existing worlds and previews if any
    print("Checking for old generated data...")
    utils.delete_files_in_path(worlds_dir)
    utils.delete_files_in_path(previews_dir)

    # print("Saving model images...")
    # keras.utils.plot_model(pconv_unet, to_file="%s\\autoencoder.png" % version_dir, show_shapes=True, show_layer_names=True)

    # Set up tensorboard
    # print("Setting up tensorboard...")
    # tb_callback = keras.callbacks.TensorBoard(log_dir=graph_dir, write_graph=True)
    # tb_callback.set_model(pconv_unet)

    # before training init writer (for tensorboard log) / model
    # tb_writer = tf.summary.FileWriter(logdir=graph_dir)
    # ae_loss = tf.Summary()
    # ae_loss.value.add(tag='ae_loss', simple_value=None)

    # Load Data
    cpu_count = multiprocessing.cpu_count()
    utilization_count = cpu_count - 1
    print("Loading worlds using %s cores." % utilization_count)
    x_train = load_worlds(world_count, "%s\\WorldRepo4\\" % cur_dir, 64, 64, minimap_values, utilization_count)

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
            world_batch_masked = utils.mask_batch(world_batch)

            loss = pconv_unet.train_on_batch(world_batch_masked, world_batch)

            test = pconv_unet.predict(world_batch_masked)

            if minibatch_index == number_of_batches - 1:
                d0 = utils.decode_world2d_binary(world_batch[0])
                utils.save_world_preview(block_images, d0, '%s\\inpainting_orig.png' % cur_dir)

                d1 = utils.decode_world2d_binary(test[0])
                utils.save_world_preview(block_images, d1, '%s\\inpainting_fixed.png' % cur_dir)

                d2 = utils.decode_world2d_binary(world_batch_masked[0][0])
                utils.save_world_preview(block_images, d2, '%s\\inpainting_masked.png' % cur_dir)

            print("loss = %f" % loss)


def main():
    train(epochs=100, batch_size=64, world_count=10000)


if __name__ == "__main__":
    main()
