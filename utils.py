from PIL import Image
from shutil import copyfile
import numpy as np
import math
import os
import gzip
import blocks
import tensorflow as tf

def load_minimap_values():
    minimapDict = {}
    with open("./Colors.txt") as fp:
        line = fp.readline()
        while line:
            spaceIndex = line.index(" ")
            keyStr = line[:spaceIndex]
            argbStr = line[spaceIndex:]
            key = int(keyStr)
            argb = int(argbStr)
            minimapDict[key] = argb
            line = fp.readline()
    return minimapDict

def load_block_images():
    blockDict = {}
    for filename in os.listdir("./blocks"):
        file = "./blocks/%s" % filename
        if os.path.isfile(file):
            blockDict[int(filename[1:-4])] = Image.open(file)
    return blockDict

def load_encoding_dict(name):
    blockForwardDict = {}
    blockBackwardDict = {}
    with open("./%s.txt" % name) as fp:
        line = fp.readline()
        blockHash = 0
        while line:
            block = int(line)
            assert block not in blockForwardDict
            blockForwardDict[block] = blockHash
            blockBackwardDict[blockHash] = block
            blockHash += 1
            line = fp.readline()
    assert len(blockForwardDict) == len(blockBackwardDict)
    return blockForwardDict, blockBackwardDict

def simplify_world(worldData):
    worldCopy = np.zeros((worldData.shape[0], worldData.shape[1]), dtype=float)
    for y in range(worldData.shape[1]):
        for x in range(worldData.shape[0]):
            worldCopy[x, y] = blocks.simplify_block(worldData[x, y])
    return worldCopy

def convert_dict_to_tensor(dict, defaultValue=0):
    dictKeys = np.asarray(list(dict.keys()))
    dictValues = np.asarray(list(dict.values()))
    dictKeysTensor = tf.cast(tf.convert_to_tensor(dictKeys), tf.int32)
    dictValuesTensor = tf.cast(tf.convert_to_tensor(dictValues), tf.int32)
    tableInitializer = tf.contrib.lookup.KeyValueTensorInitializer(dictKeysTensor, dictValuesTensor)
    tensor = tf.contrib.lookup.HashTable(tableInitializer, defaultValue)
    return tensor

def decode_world(blockBackwardDict, worldData):
    if len(worldData.shape) == 2:
        return decode_world2d(blockBackwardDict, worldData)
    elif len(worldData.shape) == 3 and worldData.shape[2] == 1:
        return decode_world2d(blockBackwardDict, np.reshape(worldData, (worldData.shape[0], worldData.shape[1])))
    elif len(worldData.shape) == 3 and worldData.shape[2] == 2:
        return decode_world3d(blockBackwardDict, worldData)
    else:
        print("Unable to decode world with shape %s" % worldData.shape)

def decode_world2d(blockBackwardDict, worldData):
    width = worldData.shape[0]
    height = worldData.shape[1]
    worldCopy = np.zeros((width, height), dtype=int)
    for y in range(height):
        for x in range(width):
            worldCopy[x, y] = decode_block(blockBackwardDict, worldData[x, y])
    return worldCopy

def decode_world3d(blockBackwardDict, worldData):
    width = worldData.shape[0]
    height = worldData.shape[1]
    worldCopy = np.zeros((width, height, 2), dtype=int)
    for y in range(height):
        for x in range(width):
            worldCopy[x, y, 0] = decode_block(blockBackwardDict, worldData[x, y, 0])
            worldCopy[x, y, 1] = decode_block(blockBackwardDict, worldData[x, y, 1])
    return worldCopy

def decode_block(blockBackwardDict, encodedValue):
    value = None
    if np.isscalar(encodedValue):
        value = encodedValue
    else:
        value = encodedValue[0]
    hashDecimal = (((value + 1) / 2) * (len(blockBackwardDict) - 1))
    truncatedHash = round(hashDecimal)
    if truncatedHash in blockBackwardDict:
        blockId = blockBackwardDict[truncatedHash]
        return blockId
    else:
        #print("Decode not in list, value is %s" % truncatedHash)
        return 0

def encode_world_softmax(blockForwardDict, worldData):
    width = worldData.shape[0]
    height = worldData.shape[1]
    worldCopy = np.zeros((width, height, 989), dtype=float)
    if len(worldData.shape) == 2:
        for y in range(height):
            for x in range(width):
                blockId = int(worldData[x, y])
                blockHash = blockForwardDict[blockId]
                worldCopy[x, y, blockHash] = 1.0
    elif len(worldData.shape) == 3: #Just take foreground
        for y in range(height):
            for x in range(width):
                blockId = int(worldData[x, y, 0])
                blockHash = blockForwardDict[blockId]
                worldCopy[x, y, blockHash] = 1.0

def encode_world(blockForwardDict, worldData):
    if len(worldData.shape) == 2:
        return encode_world2d(blockForwardDict, worldData)
    elif len(worldData.shape) == 3 and worldData.shape[2] == 1:
        return encode_world2d(blockForwardDict, np.reshape(worldData, (worldData.shape[0], worldData.shape[1])))
    elif len(worldData.shape) == 3 and worldData.shape[2] == 2:
        return encode_world3d(blockForwardDict, worldData)
    else:
        print("Unable to encode world with shape %s" % worldData.shape)

def encode_world2d(blockForwardDict, worldData):
    width = worldData.shape[0]
    height = worldData.shape[1]
    worldCopy = np.zeros((width, height, 1), dtype=float)

    if len(worldData.shape) == 2:
        for y in range(height):
            for x in range(width):
                worldCopy[x, y, 0] = encode_block(blockForwardDict, worldData[x, y])
    elif len(worldData.shape) == 3: #Just take foreground
        for y in range(height):
            for x in range(width):
                worldCopy[x, y, 0] = encode_block(blockForwardDict, worldData[x, y, 0])

    return worldCopy

def encode_world3d(blockForwardDict, worldData):
    width = worldData.shape[0]
    height = worldData.shape[1]
    worldCopy = np.zeros((width, height, 2), dtype=float)
    for y in range(height):
        for x in range(width):
            worldCopy[x, y, 0] = encode_block(blockForwardDict, worldData[x, y, 0])
            worldCopy[x, y, 1] = encode_block(blockForwardDict, worldData[x, y, 1])
    return worldCopy

def encode_block(blockForwardDict, blockId):
    if blockId not in blockForwardDict:
        return encode_block(blockForwardDict, 0)
    return ((blockForwardDict[blockId] / (len(blockForwardDict) - 1)) * 2) - 1

def encode_block_color(minimapValues, block):
    v = minimapValues[block]
    a = (v >> 24) & 0xFF;
    r = (v >> 16) & 0xFF;
    g = (v >> 8) & 0xFF;
    b = (v) & 0xFF;
    if a != 0 and b != 0 and v != 0:
        return ((r - 127.5) / 127.5, (g - 127.5) / 127.5, (b - 127.5) / 127.5)

def encode_world_minimap(minimapValues, worldData):
    if len(worldData.shape) == 2:
        return encode_world_minimap2d(minimapValues, worldData)
    elif len(worldData.shape) == 3 and worldData.shape[2] == 1:
        return encode_world_minimap2d(minimapValues, np.reshape(worldData, (worldData.shape[0], worldData.shape[1])))
    elif len(worldData.shape) == 3 and worldData.shape[2] == 2:
        return encode_world_minimap3d(minimapValues, worldData)
    else:
        print("Unable to encode world minimap with shape %s" % worldData.shape)

def encode_world_minimap2d(minimapValues, worldData):
    width = worldData.shape[0]
    height = worldData.shape[1]

    encodedValues = np.zeros((width, height, 3), dtype=float)
    for x in range(width):
        for y in range(height):
            block = int(worldData[x, y])
            if block in minimapValues:
                v = minimapValues[block]
                a = (v >> 24) & 0xFF;
                r = (v >> 16) & 0xFF;
                g = (v >> 8) & 0xFF;
                b = (v) & 0xFF;
                if a != 0 and b != 0 and v != 0:
                    encodedValues[x, y, 0] = (r - 127.5) / 127.5
                    encodedValues[x, y, 1] = (g - 127.5) / 127.5
                    encodedValues[x, y, 2] = (b - 127.5) / 127.5

    return encodedValues

def encode_world_minimap3d(minimapValues, worldData):
    width = worldData.shape[0]
    height = worldData.shape[1]

    encodedValues = np.zeros((width, height, 3), dtype=float)
    for z in range(2):
        for x in range(width):
            for y in range(height):
                block = int(worldData[x, y, 1 - z])
                if block in minimapValues:
                    v = minimapValues[block]
                    a = (v >> 24) & 0xFF;
                    r = (v >> 16) & 0xFF;
                    g = (v >> 8) & 0xFF;
                    b = (v) & 0xFF;
                    if a != 0 and b != 0 and v != 0:
                        encodedValues[x, y, 0] = (r - 127.5) / 127.5
                        encodedValues[x, y, 1] = (g - 127.5) / 127.5
                        encodedValues[x, y, 2] = (b - 127.5) / 127.5

    return encodedValues

def decode_world_minimap(minimapValues, worldData):
    width = worldData.shape[0]
    height = worldData.shape[1]

    decodedValues = np.zeros((width, height, 3), dtype=int)
    for x in range(width):
        for y in range(height):
            decodedValues[x, y, 0] = (worldData[x, y, 0] * 127.5) + 127.5
            decodedValues[x, y, 1] = (worldData[x, y, 1] * 127.5) + 127.5
            decodedValues[x, y, 2] = (worldData[x, y, 2] * 127.5) + 127.5

    return decodedValues

def save_rgb_map(rgbMap, name):
    try:
        width = rgbMap.shape[0]
        height = rgbMap.shape[1]
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        for x in range(width):
            for y in range(height):
                r = rgbMap[x, y, 0]
                g = rgbMap[x, y, 1]
                b = rgbMap[x, y, 2]
                img.putpixel((x, y), (r, g, b))
        img.save(name)
        img.close()
    except:
        print("Failed to save world minimap to %s" % name)

def load_world_data_ver2(worldFile):
    worldDataStream = gzip.open(worldFile, "rb")
    worldData = worldDataStream.readlines()
    worldDataStream.close()

    worldWidth = int(worldData[0].rstrip())
    worldHeight = int(worldData[1].rstrip())

    layerSize = worldWidth * worldHeight

    world = np.zeros((worldWidth, worldHeight, 2), dtype=float)

    for z in range(2):
        offset = (z * layerSize) + 2
        for j in range(layerSize):
            x = int(j % worldWidth)
            y = int(j / worldWidth)
            world[x, y, z] = int(worldData[offset + j].rstrip())

    return world

def load_world_data_ver3(worldFile):
    worldDataStream = gzip.open(worldFile, "r")
    worldData = worldDataStream.readline().decode("utf8").split(',')
    worldDataStream.close()
    worldWidth = int(worldData[0].rstrip())
    worldHeight = int(worldData[1].rstrip())

    layerSize = worldWidth * worldHeight

    world = np.zeros((worldWidth, worldHeight), dtype=float)

    for j in range(layerSize):
        x = int(j % worldWidth)
        y = int(j / worldWidth)
        world[x, y] = int(worldData[2 + j])
    return world

def save_world_data(worldData, name):
    try:
        f = open(name, "w")
        f.write(str(worldData.shape[0]))
        f.write('\n')
        f.write(str(worldData.shape[1]))
        f.write('\n')
        for y in range(worldData.shape[1]):
            for x in range(worldData.shape[0]):
                f.write(str(int(worldData[x, y])))
                f.write('\n')
        f.close()
    except:
        print("Failed to save world data to %s" % name)

def save_world_minimap(minimap, worldData, name):
    if len(worldData.shape) == 2:
        save_world_minimap2d(minimap, worldData, name)
    elif len(worldData.shape) == 3 and worldData.shape[2] == 1:
        save_world_minimap2d(minimap, np.reshape(worldData, (worldData.shape[0], worldData.shape[1])), name)
    elif len(worldData.shape) == 3 and worldData.shape[2] == 2:
        save_world_minimap3d(minimap, worldData, name)
    else:
        print("Unable to save minimap with shape %s" % worldData.shape)

def save_world_minimap2d(minimap, worldData, name):
    try:
        width = worldData.shape[0]
        height = worldData.shape[1]
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        for x in range(width):
            for y in range(height):
                block = int(worldData[x, y])
                if block in minimap:
                    v = minimap[block]
                    a = (v >> 24) & 0xFF;
                    r = (v >> 16) & 0xFF;
                    g = (v >> 8) & 0xFF;
                    b = (v) & 0xFF;
                    img.putpixel((x, y), (r, g, b))
        img.save(name)
        img.close()
    except:
        print("Failed to save world minimap to %s" % name)

def save_world_minimap3d(minimap, worldData, name):
    try:
        width = worldData.shape[0]
        height = worldData.shape[1]

        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        for z in range(2):
            for x in range(width):
                for y in range(height):
                    block = int(worldData[x, y, 1 - z])
                    if block in minimap:
                        v = minimap[block]
                        a = (v >> 24) & 0xFF;
                        r = (v >> 16) & 0xFF;
                        g = (v >> 8) & 0xFF;
                        b = (v) & 0xFF;
                        if a != 0 and b != 0 and v != 0:
                            img.putpixel((x, y), (r, g, b))
        img.save(name)
        img.close()
    except:
        print("Failed to save world minimap to %s" % name)

def save_world_preview(blockImages, worldData, name):
    try:
        width = worldData.shape[0]
        height = worldData.shape[1]
        img = Image.new('RGB', (width * 16, height * 16), color=(0, 0, 0))
        for x in range(width):
            for y in range(height):
                block = int(worldData[x, y])
                if block in blockImages:
                    blockImage = blockImages[block]
                    img.paste(blockImage, (x * 16, y * 16))
                else:
                    blockImage = blockImages[0]
                    img.paste(blockImage, (x * 16, y * 16))
        img.save(name, compress_level=1)
        img.close()
    except:
        print("Failed to save world minimap to %s" % name)

def shuffle_unison(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def check_or_create_local_path(name, baseDir = None):
    if baseDir is None:
        baseDir = os.getcwd()

    localDir = "%s\\%s\\" % (baseDir, name)
    if not os.path.exists(localDir):
        os.makedirs(localDir)
    return localDir

def delete_files_in_path(path):
    for filename in os.listdir(path):
        filePath = os.path.join(path, filename)
        try:
            if os.path.isfile(filePath):
                os.unlink(filePath)
            elif os.path.isdir(filePath):
                os.rmdir(filePath)
        except:
            pass

def save_source_to_dir(baseDir):
    sourceDir = check_or_create_local_path("src", baseDir)
    curDir = os.getcwd()
    for path in os.listdir(curDir):
        if os.path.isfile(path):
            copyfile(path, "%s\\%s" % (sourceDir, path))

def get_latest_version(dir):
    highestVer = 0
    for path in os.listdir(dir):
        pathVer = int(path[3:])
        if pathVer > highestVer:
            highestVer = pathVer

    return highestVer

def rotate_world90(worldData):
    worldWidth = worldData.shape[0]
    worldHeight = worldData.shape[1]

    rotatedWorld = np.zeros((worldWidth, worldHeight), dtype=int)
    for y in range(worldHeight):
        for x in range(worldWidth):
            curId = worldData[x, y]
            rotId = blocks.rotate_block(curId)

            transformedX = worldHeight - y - 1
            transformedY = x

            rotatedWorld[transformedX, transformedY] = rotId

    return rotatedWorld