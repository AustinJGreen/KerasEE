import os
import numpy as np
import time
import utils
import tests
import math
import multiprocessing
import keras
import random
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers.core import Flatten
from keras.layers import LeakyReLU, ReLU, InputLayer, Dropout
from keras.layers.noise import GaussianNoise
from keras.optimizers import Adam, SGD
from keras.models import load_model

from Minibatch import MinibatchDiscrimination
from AddMinimapValues import AddMinimapValues
from LoadWorker import GanWorldLoader
import tensorflow as tf
from keras import backend as K

def load_worlds(loadCount, worldDirectory, genWidth, genHeight, blockForwardDict, minimapValues, threadCount):
    worldNames = os.listdir(worldDirectory)
    random.shuffle(worldNames)

    with multiprocessing.Manager() as manager:
        fileQueue = manager.Queue()

        for name in worldNames:
            fileQueue.put(worldDirectory + name)

        worldArray = np.zeros((loadCount, genWidth, genHeight, 1), dtype=float)

        worldCounter = multiprocessing.Value('i', 0)
        threadLock = multiprocessing.Lock()

        threads = [None] * threadCount
        for thread in range(threadCount):
            loadThread = GanWorldLoader(fileQueue, manager, worldCounter, threadLock, loadCount, genWidth, genHeight, blockForwardDict, minimapValues)
            loadThread.start()
            threads[thread] = loadThread

        worldIndex = 0
        for thread in range(threadCount):
            threads[thread].join()

            threadLoadQueue = threads[thread].get_worlds()
            while threadLoadQueue.qsize() > 0:
                worldArray[worldIndex] = threadLoadQueue.get()
                worldIndex += 1

    return worldArray

def generator_model(blockBackwardsDict, minimapValues):
    model = Sequential(name="generator")

    # Encoder
    model.add(keras.layers.InputLayer(input_shape=(64, 64, 1)))
    model.add(AddMinimapValues(blockBackwardsDict, minimapValues))
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
    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(1, kernel_size=5, strides=2, padding="same", kernel_initializer="glorot_normal"))
    model.add(Activation("tanh"))

    model.summary()

    masked_img = Input(shape=(64, 64, 1))
    gen_missing = model(masked_img)

    return Model(masked_img, gen_missing)

def discriminator_model(blockBackwardsDict, minimapValues):
    model = Sequential(name="discriminator")

    model.add(keras.layers.InputLayer(input_shape=(64, 64, 1)))
    #model.add(AddMinimapValues(blockBackwardsDict, minimapValues))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
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
    model.add(MinibatchDiscrimination(nb_kernels=5, kernel_dim=3))
    model.add(Dense(1, activation="relu"))
    model.summary()

    masked_world = Input(shape=(64, 64, 1))
    gen_action = model(masked_world)

    return Model(masked_world, gen_action)

def generator_containing_discriminator(g, d):
    masked_world = Input(shape=(64, 64, 1))
    gen_missing = g(masked_world)
    d.trainable = False
    valid = d(gen_missing)
    combined = Model(masked_world, [gen_missing, valid])
    return combined

def train(epochs, batch_size, worldCount, versionName = None):
    curDir = os.getcwd()
    ganDir = utils.check_or_create_local_path("vac")

    if versionName is None:
        latest = utils.get_latest_version(ganDir)
        versionName = "ver%s" % (latest + 1)

    versionDir = utils.check_or_create_local_path(versionName, ganDir)
    graphDir = utils.check_or_create_local_path("graph", versionDir)
    worldsDir = utils.check_or_create_local_path("worlds", versionDir)
    previewsDir = utils.check_or_create_local_path("previews", versionDir)

    print ("Saving source...")
    utils.save_source_to_dir(versionDir)

    #Load block images
    print("Loading block images...")
    blockImages = utils.load_block_images()

    # Load minimap values
    print("Loading minimap values...")
    minimapValues = utils.load_minimap_values()

    #Load preprocessing values
    print("Loading preprocessing values...")
    blockForwardDict, blockBackwardDict = utils.load_encoding_dict("optimized2")

    #Load model and existing weights
    print("Loading model...")
    d = None
    g = None
    d_on_g = None

    #Try to load full model, otherwise try to load weights
    if os.path.exists("%s\\discriminator.h5" % versionDir) and os.path.exists("%s\\generator.h5" % versionDir):
        print("Found models.")
        d = load_model("%s\\discriminator.h5" % versionDir)
        g = load_model("%s\\generator.h5" % versionDir)
        d_on_g = generator_containing_discriminator(g, d)
    else:
        #Load any existing weights if any
        if os.path.exists("%s\\discriminator.model" % versionDir) and os.path.exists("%s\\generator.model" % versionDir):
            print("Found weights.")
            d.load_weights("%s\\discriminator.model" % versionDir)
            g.load_weights("%s\\generator.model" % versionDir)

        print("Compiling model...")
        optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

        d = discriminator_model(blockBackwardDict, minimapValues)
        d.compile(loss="binary_crossentropy", optimizer=optim, metrics=["accuracy"])

        g = generator_model(blockBackwardDict, minimapValues)
        d_on_g = generator_containing_discriminator(g, d)

        d_on_g.compile(loss=["mse", "binary_crossentropy"], loss_weights=[0.999, 0.001], optimizer=optim)

    #Delete existing worlds and previews if any
    print("Checking for old generated data...")
    utils.delete_files_in_path(worldsDir)
    utils.delete_files_in_path(previewsDir)

    #print("Saving model images...")
    keras.utils.plot_model(d, to_file="%s\\discriminator.png" % versionDir, show_shapes=True, show_layer_names=True)
    keras.utils.plot_model(g, to_file="%s\\generator.png" % versionDir, show_shapes=True, show_layer_names=True)

    #Set up tensorboard
    print("Setting up tensorboard...")
    tbCallback = keras.callbacks.TensorBoard(log_dir=graphDir, write_graph=True)
    tbCallback.set_model(d_on_g)

    # before training init writer (for tensorboard log) / model
    tbWriter = tf.summary.FileWriter(logdir=graphDir)
    dAccSummary = tf.Summary()
    dAccSummary.value.add(tag='d_acc', simple_value=None)
    dLossSummary = tf.Summary()
    dLossSummary.value.add(tag='d_loss', simple_value=None)
    gLossSummary = tf.Summary()
    gLossSummary.value.add(tag='g_loss', simple_value=None)

    # Load Data
    cpuCount = multiprocessing.cpu_count()
    utilizationCount = cpuCount - 1
    print ("Loading worlds using %s cores." % (utilizationCount))
    X_train = load_worlds(worldCount, "%s\\WorldRepo3\\" % curDir, 64, 64, blockForwardDict, minimapValues, utilizationCount)

    #Start Training loop
    numberOfBatches = int(worldCount / batch_size)

    #Initialize tables for Hashtable tensors
    K.get_session().run(tf.tables_initializer())

    for epoch in range(epochs):

        #Create directories for current epoch
        curWorldsCur = utils.check_or_create_local_path("epoch%s" % epoch, worldsDir)
        curPreviewsDir = utils.check_or_create_local_path("epoch%s" % epoch, previewsDir)

        print("Shuffling data...")
        np.random.shuffle(X_train)

        for minibatchIndex in range(numberOfBatches):

            # Get real set of images
            realWorlds = X_train[minibatchIndex * batch_size:(minibatchIndex + 1) * batch_size]

            #Get fake set of images
            fakeWorlds = g.predict(realWorlds)

            realLabels = np.ones((batch_size, 1))  # np.random.uniform(0.9, 1.1, size=(batch_size,))
            fakeLabels = np.zeros((batch_size, 1))  # np.random.uniform(-0.1, 0.1, size=(batch_size,))

            #Save snapshot of generated images on last batch
            if minibatchIndex == numberOfBatches - 1:
                for batchImage in range(batch_size):
                    generatedWorld = fakeWorlds[batchImage]
                    decodedWorld = utils.decode_world(blockBackwardDict, generatedWorld)
                    utils.save_world_data(decodedWorld, "%s\\world%s.dat" % (curWorldsCur, batchImage))
                    utils.save_world_preview(blockImages, decodedWorld, "%s\\preview%s.png" % (curPreviewsDir, batchImage))

            #Train discriminator on real worlds
            d.trainable = True
            d_loss = d.train_on_batch(realWorlds, realLabels)
            dAccSummary.value[0].simple_value = d_loss[1]
            tbWriter.add_summary(dAccSummary, (epoch * numberOfBatches) + minibatchIndex)
            dLossSummary.value[0].simple_value = d_loss[0]
            tbWriter.add_summary(dLossSummary, (epoch * numberOfBatches) + minibatchIndex)

            #Train discriminator on fake worlds
            d_loss = d.train_on_batch(fakeWorlds, fakeLabels)
            dAccSummary.value[0].simple_value = d_loss[1]
            tbWriter.add_summary(dAccSummary, (epoch * numberOfBatches) + minibatchIndex)
            dLossSummary.value[0].simple_value = d_loss[0]
            tbWriter.add_summary(dLossSummary, (epoch * numberOfBatches) + minibatchIndex)
            d.trainable = False

            #Train generator to generate real
            g_loss = d_on_g.train_on_batch(realWorlds, [realWorlds, realLabels])
            gLossSummary.value[0].simple_value = g_loss[0]
            tbWriter.add_summary(gLossSummary, (epoch * numberOfBatches) + minibatchIndex)
            tbWriter.flush()

            print("epoch [%d/%d] :: batch [%d/%d] :: disAcc = %.1f%% :: disLoss = %f :: genLoss = %f" % (epoch, epochs, minibatchIndex, numberOfBatches, d_loss[1] * 100, d_loss[0], g_loss[0]))

            #Save models
            if minibatchIndex % 100 == 99 or minibatchIndex == numberOfBatches - 1:
                d.save("%s\\discriminator.h5" % versionDir)
                g.save("%s\\generator.h5" % versionDir)

def main():
    #tests.test_addminimap_layer()
    train(epochs=1000, batch_size=32, worldCount=5000)

if __name__ == "__main__":
    main()

