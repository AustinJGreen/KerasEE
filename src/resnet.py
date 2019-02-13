from keras.layers import Dense, Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dropout
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model


def identity_block(input_tensor, kernel_size, filters, stage, block, freeze=False, dropout=True):
    train = not freeze
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2a', trainable=train)(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a', trainable=train)(x)
    x = Activation('relu', trainable=train)(x)

    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2b', trainable=train)(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b', trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    if dropout:
        x = Dropout(0.1)(x)

    x = Conv2D(filters3, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2c', trainable=train)(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c', trainable=train)(x)

    x = Add(trainable=train)([x, input_tensor])
    x = Activation('relu', trainable=train)(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               freeze=False, dropout=True):
    train = not freeze
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               kernel_initializer='he_normal',
               name=conv_name_base + '2a', trainable=train)(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a', trainable=train)(x)
    x = Activation('relu', trainable=train)(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2b', trainable=train)(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b', trainable=train)(x)
    x = Activation('relu', trainable=train)(x)
    if dropout:
        x = Dropout(0.1)(x)

    x = Conv2D(filters3, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2c', trainable=train)(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c', trainable=train)(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '1', trainable=train)(input_tensor)
    shortcut = BatchNormalization(
        axis=3, name=bn_name_base + '1', trainable=train)(shortcut)

    x = Add(trainable=train)([x, shortcut])
    x = Activation('relu', trainable=train)(x)
    return x


def build_resnet50(classes):
    input_tensor = Input(shape=(112, 112, 10))

    x = Conv2D(64, (7, 7),
               strides=(2, 2),
               padding='valid',
               kernel_initializer='he_normal',
               name='conv1', trainable=True)(input_tensor)
    x = BatchNormalization(axis=3, name='bn_conv1', trainable=True)(x)
    x = Activation('relu', trainable=True)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), trainable=True)(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), freeze=False)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', freeze=False)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', freeze=False)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', freeze=False)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', freeze=False)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', freeze=False)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', freeze=False)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', freeze=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', freeze=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', freeze=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', freeze=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', freeze=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', freeze=False)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', freeze=False)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', freeze=False)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', freeze=False)

    x = GlobalAveragePooling2D(name='avg_pool', trainable=True)(x)

    if classes > 1:
        x = Dense(units=classes, activation='softmax', name='fc3', trainable=True)(x)
    else:
        x = Dense(units=1, activation='sigmoid', name='fc3', trainable=True)(x)

    # Create model
    model = Model(inputs=[input_tensor], outputs=[x], name='resnet50')
    return model


def build_resnet18(classes):
    input_tensor = Input(shape=(112, 112, 10))

    x = Conv2D(64, (7, 7),
               strides=(2, 2),
               padding='valid',
               kernel_initializer='he_normal',
               name='conv1', trainable=True)(input_tensor)
    x = BatchNormalization(axis=3, name='bn_conv1', trainable=True)(x)
    x = Activation('relu', trainable=True)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), trainable=True)(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), freeze=False)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', freeze=False)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', freeze=False)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', freeze=False)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', freeze=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', freeze=False)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', freeze=False)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', freeze=False)

    x = GlobalAveragePooling2D(name='avg_pool', trainable=True)(x)

    if classes > 1:
        x = Dense(units=classes, activation='softmax', name='fc3', trainable=True)(x)
    else:
        x = Dense(units=1, activation='sigmoid', name='fc3', trainable=True)(x)

    # Create model
    model = Model(inputs=[input_tensor], outputs=[x], name='resnet18')
    return model
