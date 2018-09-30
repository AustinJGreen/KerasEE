import keras
from keras.models import Sequential
from AddMinimapValues import AddMinimapValues
import tensorflow as tf
from keras import backend as K
import utils

def test_addminimap_layer():
    minimapValues = utils.load_minimap_values()
    blockForwardDict, blockBackwardsDict = utils.load_encoding_dict("optimized2")

    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=(64, 64, 1)))
    model.add(AddMinimapValues(blockBackwardsDict, minimapValues))
    model.compile(loss='binary_crossentropy', optimizer="sgd")

    rawWorld = utils.load_world_data_ver3("./WorldRepo3/PW-pATu-3za0I.world")[0:64,0:64]
    utils.save_world_minimap(minimapValues, rawWorld, "C:\\Users\\austi\\Desktop\\orig.png")

    encodedWorld = utils.encode_world2d(blockForwardDict, rawWorld)
    batch = encodedWorld[None, :, :, :]

    K.get_session().run(tf.tables_initializer())
    output = model.predict(batch)

    outputWorld = output[0]
    worldMinimap = outputWorld[:, :, 1:4]
    worldRgbMap = utils.decode_world_minimap(minimapValues, worldMinimap)
    utils.save_rgb_map(worldRgbMap, "C:\\Users\\austi\\Desktop\\feed.png")

    print("Done")

def test_mapping():
    blockForwardDict, blockBackwardDict = utils.load_encoding_dict("optimized")
    testValue = utils.encode_block(blockForwardDict, 9)
    inverseValue = utils.decode_block(blockBackwardDict, testValue)
    assert inverseValue == 9

def test_encoding():
    blockForwardDict, blockBackwardDict = utils.load_encoding_dict("optimized")
    encodingTest1 = utils.encode_block(blockForwardDict, 9)
    encodingTest2 = utils.encode_block(blockForwardDict, 100)
    encodingTest3 = utils.encode_block(blockForwardDict, 107)

    decodingTest0 = utils.decode_block(blockBackwardDict, encodingTest3)

    decodingTest1 = utils.decode_block(blockBackwardDict, encodingTest1)
    decodingTest2 = utils.decode_block(blockBackwardDict, encodingTest2)

    decodingTest3 = utils.decode_block(blockBackwardDict, 0.004)
    decodingTest4 = utils.decode_block(blockBackwardDict, 0.005)
    return

def test_rotations():
    rawWorld = utils.load_world_data_ver3("./WorldRepo3/PW-pATu-3za0I.world")[0:64, 0:64]
    blockImages = utils.load_block_images()
    utils.save_world_preview(blockImages, rawWorld, "C:\\Users\\austi\\Desktop\\raw.png")

    rotated90 = utils.rotate_world90(rawWorld)
    utils.save_world_preview(blockImages, rotated90, "C:\\Users\\austi\\Desktop\\rot90.png")

    rotated180 = utils.rotate_world90(rotated90)
    utils.save_world_preview(blockImages, rotated180, "C:\\Users\\austi\\Desktop\\rot180.png")

    rotated270 = utils.rotate_world90(rotated180)
    utils.save_world_preview(blockImages, rotated270, "C:\\Users\\austi\\Desktop\\rot270.png")

    rotated360 = utils.rotate_world90(rotated270)
    utils.save_world_preview(blockImages, rotated360, "C:\\Users\\austi\\Desktop\\rot360.png")