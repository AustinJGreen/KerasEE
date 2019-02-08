from keras.layers import Dense, Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model


def identity_block(input_tensor, kernel_size, filters, stage, block, freeze=False):
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
               freeze=False):
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


def build_resnet50():
    input_tensor = Input(shape=(112, 112, 10))

    x = Conv2D(64, (7, 7),
               strides=(2, 2),
               padding='valid',
               kernel_initializer='he_normal',
               name='conv1', trainable=False)(input_tensor)
    x = BatchNormalization(axis=3, name='bn_conv1', trainable=False)(x)
    x = Activation('relu', trainable=False)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), trainable=False)(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), freeze=True)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', freeze=True)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', freeze=True)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', freeze=True)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', freeze=True)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', freeze=True)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', freeze=True)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', freeze=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', freeze=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', freeze=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', freeze=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', freeze=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', freeze=False)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', freeze=True)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', freeze=True)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', freeze=True)

    x = GlobalAveragePooling2D(name='avg_pool', trainable=False)(x)
    x = Dense(units=3, activation='softmax', name='fc3', trainable=False)(x)

    # Create model
    model = Model(inputs=[input_tensor], outputs=[x], name='resnet50')
    return model
