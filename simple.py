#System imports
import os
import numpy as np

import blocks
import utils

#Keras Imports
import keras
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.optimizers import Adam

def remove_action(worldData, width, height):
    worldCopy = np.zeros((width, height), dtype=int)
    for y in range(height):
        for x in range(width):
            worldCopy[x, y] = worldData[x, y]
            if blocks.is_action(worldCopy[x, y]):
                worldCopy[x, y] = 0
    return worldCopy

def load_world(X_train, Y_train, worldsLoaded, worldFile, genWidth, genHeight, blockForwardDict):
    worldDataStream = open(worldFile, "r")
    worldData = worldDataStream.readlines()
    worldDataStream.close()

    worldWidth = int(worldData[0].rstrip())
    worldHeight = int(worldData[1].rstrip())

    world = np.zeros((worldWidth, worldHeight), dtype=float)

    j = 2
    while j < len(worldData) - 1:
        x = int((j - 2) % worldWidth)
        y = int((j - 2) / worldWidth)
        world[x, y] = int(worldData[j].rstrip())
        j = j + 1

    xMargin = worldWidth % genWidth
    yMargin = worldHeight % genHeight

    horizontalCrossSections = int((worldWidth - xMargin) / genWidth)
    verticalCrossSections = int((worldHeight - yMargin) / genHeight)

    xOffset = 0
    yOffset = 0

    if xMargin > 0:
        xOffset = np.random.randint(0, xMargin)
        yOffset = np.random.randint(0, yMargin)


    for yCrossSection in range(verticalCrossSections):
        for xCrossSection in range(horizontalCrossSections):
            xStart = xOffset + (xCrossSection * genWidth)
            xEnd = xStart + genWidth
            yStart = yOffset + (yCrossSection * genHeight)
            yEnd = yStart + genHeight
            worldData = world[xStart:xEnd, yStart:yEnd]

            # Preprocess world
            xWorldData = remove_action(worldData, genWidth, genHeight)

            #Validify
            xBlockCnt = 0
            for y in range(genHeight):
                for x in range(genWidth):
                    if xWorldData[x, y] != 0:
                        xBlockCnt += 1

            #Check to make sure data has some blocks
            if xBlockCnt >= ((genWidth * genHeight) * 0.25):
                xEncoded = utils.encode_world(blockForwardDict, xWorldData)
                X_train[worldsLoaded] = xEncoded

                yEncoded = utils.encode_world(blockForwardDict, worldData)
                Y_train[worldsLoaded] = yEncoded
                worldsLoaded += 1

                if worldsLoaded >= len(X_train):
                    return len(X_train)

    return worldsLoaded

def load_worlds(loadCount, worldDirectory, genWidth, genHeight, blockForwardDict):
    worldNames = os.listdir(worldDirectory)

    worldsLoaded = 0
    X_train = np.zeros((loadCount, genWidth, genHeight), dtype=float)
    Y_train = np.zeros((loadCount, genWidth, genHeight), dtype=float)
    for name in worldNames:
        worldsLoaded = load_world(X_train, Y_train, worldsLoaded, worldDirectory + name, genWidth, genHeight, blockForwardDict)
        print("Loaded (%s/%s)" % (worldsLoaded, loadCount))
        if worldsLoaded >= loadCount:
            break

    return X_train, Y_train

def build_generator():
    model = Sequential()

    # Encoder
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(64, 64, 1), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(512, kernel_size=1, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))

    # Decoder
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(1, kernel_size=3, padding="same"))
    model.add(Activation('tanh'))

    model.summary()

    masked_img = Input(shape=(64, 64, 1))
    gen_missing = model(masked_img)

    return Model(masked_img, gen_missing)

def build_discriminator():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(64, 64, 1), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    model.summary()

    masked_world = Input(shape=(64, 64, 1))
    gen_action = model(masked_world)

    return Model(masked_world, gen_action)

def train(epochs, batch_size, versionName):
    curDirectory = os.getcwd()

    #Create simple directory
    simpleDir = "%s\\%s" % (curDirectory, "simple")
    if not os.path.exists(simpleDir):
        os.makedirs(simpleDir)

    # Create version directory
    versionDir = "%s\\%s\\" % (simpleDir, versionName)
    if not os.path.exists(versionDir):
        os.makedirs(versionDir)

    graphDir = "%s\\%s\\Graph" % (simpleDir, versionName)
    if not os.path.exists(graphDir):
        os.makedirs(graphDir)

    #Load model
    print("Loading model...")
    optimizer = keras.optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
    discriminator = build_discriminator()
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    generator = build_generator()
    masked_world = Input(shape=(64, 64, 1))
    gen_missing = generator(masked_world)

    discriminator.trainable = False
    valid = discriminator(gen_missing)

    combined = Model(masked_world, [gen_missing, valid])
    combined.compile(loss=["mse", "binary_crossentropy"], loss_weights=[0.999, 0.001], optimizer=optimizer)

    print("Saving model...")
    keras.utils.plot_model(generator, to_file="%s\\%s\\generator.png" % (simpleDir, versionName), show_shapes=True,
                           show_layer_names=True)
    keras.utils.plot_model(discriminator, to_file="%s\\%s\\discriminator.png" % (simpleDir, versionName), show_shapes=True,
                           show_layer_names=True)

    #Set up tensorboard
    print("Setting up tensorboard...")
    tbCallback = keras.callbacks.TensorBoard(log_dir=graphDir, histogram_freq=1, write_graph=True)
    tbCallback.set_model(combined)

    #Load preprocessing values
    print("Loading preprocessing values...")
    blockForwardDict, blockBackwardDict = utils.load_encoding_dict("Map1")

    #Quick tests
    testValue = utils.encode_block(blockForwardDict, 9)
    inverseValue = utils.decode_block(blockBackwardDict, [testValue])
    assert inverseValue == 9

    #testA = np.asarray([1, 2, 3, 4])
    #testB = np.asarray([11, 12, 13, 14])
    #shuff = utils.shuffle_unison(testA, testB)

    # Load Data
    X_train, Y_train = load_worlds(30000, "%s\\WorldRepo1\\" % curDirectory, 64, 64, blockForwardDict)
    utils.shuffle_unison(X_train, Y_train)

    X_train = X_train[:, :, :, None]
    Y_train = Y_train[:, :, :, None]

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    #Start Training loop
    numberOfBatches = int(X_train.shape[0] / batch_size)
    for epoch in range(epochs):
        # Create world directory
        worldDataDir = "%s\\%s\\worlds\\epoch%s\\" % (simpleDir, versionName, epoch)
        if not os.path.exists(worldDataDir):
            os.makedirs(worldDataDir)

        # Create minimap directory
        '''
        minimapDir = "%s\\%s\\minimaps\\epoch%s\\" % (simpleDir, versionName, epoch)
        if not os.path.exists(minimapDir):
            os.makedirs(minimapDir)
        '''

        # Create preview directory
        previewDir = "%s\\%s\\previews\\epoch%s\\" % (simpleDir, versionName, epoch)
        if not os.path.exists(previewDir):
            os.makedirs(previewDir)

        for index in range(numberOfBatches):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            missingWorlds = X_train[index * batch_size:(index + 1) * batch_size]
            realWorlds = Y_train[index * batch_size:(index + 1) * batch_size]

            # Generate a batch of new images
            gen_missing = generator.predict(realWorlds)

            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(missingWorlds, valid)
            d_loss_fake = discriminator.train_on_batch(gen_missing, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) #Average loss

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = combined.train_on_batch(realWorlds, [realWorlds, valid])

            # Plot the progress
            print("(%d/%d) (%d/%d) [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
            epoch, epochs, index, numberOfBatches, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated image samples
            if index == numberOfBatches - 1:
                # Current set of generated images
                for batchImage in range(batch_size):
                    generatedWorld = gen_missing[batchImage]

                    #Deprocess the world
                    decodedWorld = utils.decode_world(blockBackwardDict, generatedWorld)

                    utils.save_world_data(decodedWorld, "%s\\%s\\worlds\\epoch%s\\world%s.dat" % (simpleDir, versionName, epoch, batchImage))
                    utils.save_world_preview(decodedWorld, "%s\\%s\\previews\\epoch%s\\preview%s.png" % (simpleDir, versionName, epoch, batchImage))
                    #save_world_minimap(minimap, decodedWorld, "%s\\%s\\minimaps\\epoch%s\\minimap%s.png" % (curDirectory, versionName, epoch, batchImage))


def main():
    train(50, 128, "ver23")

if __name__ == "__main__":
    main()