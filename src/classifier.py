import os

import keras
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

import utils
from loadworker import load_worlds_with_labels


def build_classifier():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=5, strides=1, padding='same', input_shape=(128, 128, 10)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=5, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))  # 64 x 64

    model.add(Conv2D(filters=128, kernel_size=5, strides=1, padding='same', input_shape=(64, 64, 10)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=128, kernel_size=5, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=128, kernel_size=5, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))  # 32 x 32

    model.add(Conv2D(filters=256, kernel_size=5, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=256, kernel_size=5, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))  # 16 x 16

    model.add(Conv2D(filters=512, kernel_size=5, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=512, kernel_size=5, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))  # 8 x 8

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    model.summary()
    return model


def train(epochs, batch_size, world_count, version_name=None, initial_epoch=0):
    cur_dir = os.getcwd()
    res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
    all_models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
    model_dir = utils.check_or_create_local_path("classifier", all_models_dir)

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

    print("Loading block images...")
    block_images = utils.load_block_images(res_dir)

    print("Loading encoding dictionaries...")
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'optimized')

    print("Building model from scratch...")
    c_optim = Adam(lr=0.0001)

    c = build_classifier()
    c.compile(loss="binary_crossentropy", optimizer=c_optim, metrics=["accuracy"])

    print("Loading labels...")
    label_dict = utils.load_label_dict(res_dir, 'world_labels_basic')

    print("Loading worlds...")
    x_train, y_labels = load_worlds_with_labels(world_count, '%s\\worlds\\' % res_dir, label_dict, (128, 128),
                                                block_forward,
                                                utils.encode_world_sigmoid, overlap_x=0.5, overlap_y=0.5)

    y_train = utils.convert_labels(y_labels, categories=3, epsilon=0)

    # Create callback for automatically saving best model based on highest regular accuracy
    check_best_acc = keras.callbacks.ModelCheckpoint('%s\\best_acc.h5' % model_save_dir, monitor='acc', verbose=0,
                                                     save_best_only=True, save_weights_only=False, mode='max',
                                                     period=1)

    # Create callback for automatically saving best model based on highest validation accuracy
    check_best_val_acc = keras.callbacks.ModelCheckpoint('%s\\best_val_acc.h5' % model_save_dir, monitor='val_acc',
                                                         verbose=0,
                                                         save_best_only=True, save_weights_only=False, mode='max',
                                                         period=1)

    # Create callback for automatically saving best model base on lowest validation loss
    check_best_val_loss = keras.callbacks.ModelCheckpoint('%s\\best_val_loss.h5' % model_save_dir, monitor='val_loss',
                                                          verbose=0,
                                                          save_best_only=True, save_weights_only=False, mode='min',
                                                          period=1)

    # Create periodic checkmark model every 50 epochs in case we need to revert back from latest
    checkpoint_callback = keras.callbacks.ModelCheckpoint('%s\\epoch{epoch:03d}.h5' % model_save_dir, verbose=0,
                                                          save_best_only=False, mode='auto', period=5)

    # Create callback for automatically saving lastest model so training can be resumed. Saves every epoch
    check_latest_callback = keras.callbacks.ModelCheckpoint('%s\\latest.h5' % model_save_dir, verbose=0,
                                                            save_best_only=False,
                                                            save_weights_only=False, mode='auto', period=1)

    # Create callback for tensorboard
    tb_callback = keras.callbacks.TensorBoard(log_dir=graph_version_dir, batch_size=batch_size, write_graph=False,
                                              write_grads=True)

    c.fit(x_train, y_train, batch_size, epochs, validation_split=0.2, callbacks=[tb_callback])


def main():
    train(epochs=100, batch_size=16, world_count=10000)


if __name__ == "__main__":
    main()
