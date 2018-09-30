import keras
from keras import backend as K
import numpy as np
import tensorflow as tf
import utils

class AddMinimapValues(keras.layers.Layer):

    minimapValuesTensor = None
    backwardsDictTensor = None
    blockCountTensor = None
    rgbConst = None
    bitMask = None

    def __init__(self, blockBackwardsDict, minimapValues, **kwargs):
        self.blockCountTensor = tf.convert_to_tensor(tf.constant(float(len(blockBackwardsDict))))
        self.backwardsDictTensor = utils.convert_dict_to_tensor(blockBackwardsDict)
        self.minimapValuesTensor = utils.convert_dict_to_tensor(minimapValues)
        self.rgbConst = tf.constant(127.5, dtype=tf.float32)
        self.bitMask = tf.constant(0xFF, dtype=tf.int32)
        self.output_dim = (None, 64, 64, 4)
        super(AddMinimapValues, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (None, 64, 64, 4)

    def decode_block_tensor(self, encodedValue):
        hashDecimal = (((encodedValue + 1) / 2) * (self.blockCountTensor - 1))
        truncatedHash = tf.cast(tf.round(hashDecimal), dtype=tf.int32)
        blockIdTensor = self.backwardsDictTensor.lookup(truncatedHash)  # 0 if not found (set as default in table initializer)
        return blockIdTensor

    def encode_block_color_red(self, minimapValuesTensor, blockTensor):
        v = minimapValuesTensor.lookup(tf.cast(blockTensor, dtype=tf.int32))
        r = tf.bitwise.bitwise_and(tf.bitwise.right_shift(v, tf.constant(16, dtype=tf.int32)), self.bitMask)
        rTensor = tf.divide(tf.subtract(tf.cast(r, dtype=tf.float32), self.rgbConst), self.rgbConst)
        return rTensor

    def encode_block_color_green(self, minimapValuesTensor, blockTensor):
        v = minimapValuesTensor.lookup(tf.cast(blockTensor, dtype=tf.int32))
        g = tf.bitwise.bitwise_and(tf.bitwise.right_shift(v, tf.constant(8, dtype=tf.int32)), self.bitMask)
        gTensor = tf.divide(tf.subtract(tf.cast(g, dtype=tf.float32), self.rgbConst), self.rgbConst)
        return gTensor

    def encode_block_color_blue(self, minimapValuesTensor, blockTensor):
        v = minimapValuesTensor.lookup(tf.cast(blockTensor, dtype=tf.int32))
        b = tf.bitwise.bitwise_and(v, self.bitMask)
        bTensor = tf.divide(tf.subtract(tf.cast(b, dtype=tf.float32), self.rgbConst), self.rgbConst)
        return bTensor

    def get_world_red_layer(self, blocksTensor):
        return self.encode_block_color_red(self.minimapValuesTensor, blocksTensor)

    def get_world_green_layer(self, blocksTensor):
        return self.encode_block_color_green(self.minimapValuesTensor, blocksTensor)

    def get_world_blue_layer(self, blocksTensor):
        return self.encode_block_color_blue(self.minimapValuesTensor, blocksTensor)

    def call(self, x):
        blocksTensor = self.decode_block_tensor(x)
        redLayer = self.get_world_red_layer(blocksTensor)
        greenLayer = self.get_world_green_layer(blocksTensor)
        blueLayer = self.get_world_blue_layer(blocksTensor)
        combinedLayer = K.concatenate([x, redLayer, greenLayer, blueLayer], axis=3)
        return combinedLayer