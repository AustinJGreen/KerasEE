import os

import keras
from keras.layers import Dense, SpatialDropout2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam

import utils
from loadworker import load_worlds_with_labels, load_worlds_with_files


def build_classifier(size):
    model = Sequential()

    f = 64
    s = size

    while s > 7:
        if s == size:
            model.add(Conv2D(filters=f, kernel_size=5, strides=1, padding='same', input_shape=(size, size, 10)))
        else:
            model.add(Conv2D(filters=f, kernel_size=5, strides=1, padding='same'))

        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(Conv2D(filters=f, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(SpatialDropout2D(0.2))

        f = f * 2
        s = s // 2

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    return model


def train(epochs, batch_size, world_count, dict_src_name, version_name=None, initial_epoch=0):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    all_models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path("pro_classifier", all_models_dir)

    utils.delete_empty_versions(model_dir, 1)
    no_version = version_name is None
    if no_version:
        latest = utils.get_latest_version(model_dir)
        version_name = "ver%s" % (latest + 1)

    version_dir = utils.check_or_create_local_path(version_name, model_dir)
    graph_dir = utils.check_or_create_local_path("graph", model_dir)
    graph_version_dir = utils.check_or_create_local_path(version_name, graph_dir)

    model_save_dir = utils.check_or_create_local_path("models", version_dir)

    print("Saving source...")
    utils.save_source_to_dir(version_dir)

    print("Loading encoding dictionaries...")
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'blocks_optimized')

    print("Building model from scratch...")
    c_optim = Adam(lr=0.0001)

    c = build_classifier(112)
    # c = build_resnet18(1)

    c.summary()
    c.compile(loss="binary_crossentropy", optimizer=c_optim, metrics=["accuracy"])

    print("Loading labels...")
    label_dict = utils.load_label_dict(res_dir, dict_src_name)

    print("Loading worlds...")
    x_train, y_labels = load_worlds_with_labels(world_count, '%s\\worlds\\' % res_dir, label_dict, (112, 112),
                                                block_forward)

    y_train = utils.convert_labels_binary(y_labels, epsilon=0)

    # Create callback for automatically saving best model based on highest regular accuracy
    check_best_acc = keras.callbacks.ModelCheckpoint('%s\\best_acc.h5' % model_save_dir, monitor='acc', verbose=0,
                                                     save_best_only=True, save_weights_only=False, mode='max',
                                                     period=1)

    # Create callback for automatically saving best model based on highest validation accuracy
    check_best_val_acc = keras.callbacks.ModelCheckpoint('%s\\best_val_acc.h5' % model_save_dir, monitor='val_acc',
                                                         verbose=0,
                                                         save_best_only=True, save_weights_only=False, mode='max',
                                                         period=1)

    # Create callback for automatically saving lastest model so training can be resumed. Saves every epoch
    latest_h5_callback = keras.callbacks.ModelCheckpoint('%s\\latest.h5' % model_save_dir, verbose=0,
                                                         save_best_only=False,
                                                         save_weights_only=False, mode='auto', period=1)

    # Create callback for automatically saving lastest weights so training can be resumed. Saves every epoch
    latest_weights_callback = keras.callbacks.ModelCheckpoint('%s\\latest.weights' % model_save_dir, verbose=0,
                                                              save_best_only=False,
                                                              save_weights_only=True, mode='auto', period=1)

    # Create callback for tensorboard
    tb_callback = keras.callbacks.TensorBoard(log_dir=graph_version_dir, batch_size=batch_size, write_graph=False,
                                              write_grads=True)

    callback_list = [check_best_acc, latest_h5_callback, latest_weights_callback, tb_callback, check_best_val_acc]

    c.fit(x_train, y_train, batch_size, epochs, callbacks=callback_list, validation_split=0.2)


def predict(network_ver, dict_src_name):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    all_models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path("pro_classifier", all_models_dir)
    version_dir = utils.check_or_create_local_path(network_ver, model_dir)
    model_save_dir = utils.check_or_create_local_path("models", version_dir)

    classifications_dir = utils.check_or_create_local_path('classifications', model_dir)
    utils.delete_files_in_path(classifications_dir)

    pro_dir = utils.check_or_create_local_path('pro', classifications_dir)
    notpro_dir = utils.check_or_create_local_path('notpro', classifications_dir)

    print("Loading model...")
    classifier = load_model('%s\\latest.h5' % model_save_dir)

    print("Loading block images...")
    block_images = utils.load_block_images(res_dir)

    print("Loading encoding dictionaries...")
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'blocks_optimized')

    x_data, x_files = load_worlds_with_files(5000, '%s\\worlds\\' % res_dir, (112, 112), block_forward)

    x_labeled = utils.load_label_dict(res_dir, dict_src_name)

    batch_size = 50
    batches = x_data.shape[0] // batch_size

    for batch_index in range(batches):
        x_batch = x_data[batch_index * batch_size:(batch_index + 1) * batch_size]
        y_batch = classifier.predict(x_batch)

        for world in range(batch_size):
            g_index = (batch_index * batch_size) + world
            world_file = x_files[g_index]
            world_id = utils.get_world_id(world_file)

            # Ignore worlds we've already labeled
            if world_id in x_labeled:
                continue

            prediction = y_batch[world]

            world_data = utils.load_world_data_ver3('%s\\worlds\\%s.world' % (res_dir, world_id))

            if prediction[0] < 0.5:
                utils.save_world_preview(block_images, world_data, '%s\\%s.png' % (notpro_dir, world_id))
            else:
                utils.save_world_preview(block_images, world_data, '%s\\%s.png' % (pro_dir, world_id))


def add_training_data(current_label_dict):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    all_models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path("pro_classifier", all_models_dir)
    classifications_dir = utils.check_or_create_local_path('classifications', model_dir)
    pro_dir = utils.check_or_create_local_path('pro', classifications_dir)
    notpro_dir = utils.check_or_create_local_path('notpro', classifications_dir)

    notpro_worlds = os.listdir(notpro_dir)
    pro_worlds = os.listdir(pro_dir)

    orig_dict = utils.load_label_dict(res_dir, current_label_dict)
    new_dict = utils.load_label_dict(res_dir, current_label_dict)

    for notpro_world in notpro_worlds:
        notpro_id = utils.get_world_id(notpro_world)
        if notpro_id not in orig_dict:
            new_dict[notpro_id] = 0

    for pro_world in pro_worlds:
        pro_id = utils.get_world_id(pro_world)
        new_dict[pro_id] = 1

    utils.save_label_dict(classifications_dir, 'test', new_dict)


def main():
    train(epochs=50, batch_size=16, world_count=25000, dict_src_name='pro_labels_b')
    # predict('ver9', dict_src_name='pro_labels')
    # add_training_data('pro_labels')


if __name__ == "__main__":
    main()