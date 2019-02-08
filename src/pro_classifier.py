import os

import keras
from keras.layers import Dense, SpatialDropout2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

import utils
from loadworker import load_worlds_with_labels


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
        model.add(SpatialDropout2D(0.3))

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
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'optimized')

    print("Building model from scratch...")
    c_optim = Adam(lr=0.0001)

    c = build_classifier(112)

    c.summary()
    c.compile(loss="binary_crossentropy", optimizer=c_optim, metrics=["accuracy"])

    print("Loading labels...")
    label_dict = utils.load_label_dict(res_dir, dict_src_name)

    print("Loading worlds...")
    x_train, y_labels = load_worlds_with_labels(world_count, '%s\\worlds\\' % res_dir, label_dict, (112, 112),
                                                block_forward,
                                                utils.encode_world_sigmoid, overlap_x=1, overlap_y=1)

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


def main():
    train(epochs=50, batch_size=100, world_count=25000, dict_src_name='pro_labels')


if __name__ == "__main__":
    main()
