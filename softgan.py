import os
import numpy as np
import time
import utils
import tests
import math
import multiprocessing
import keras
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.layers.core import Flatten
from keras.layers import LeakyReLU, ReLU, InputLayer, Dropout, Softmax
from keras.layers.noise import GaussianNoise
from keras.optimizers import Adam, SGD
from keras.models import load_model

from Minibatch import MinibatchDiscrimination
from AddMinimapValues import AddMinimapValues
from LoadWorker import SoftGanWorldLoader
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
            loadThread = SoftGanWorldLoader(fileQueue, manager, worldCounter, threadLock, loadCount, genWidth, genHeight, blockForwardDict, minimapValues)
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

def check_grads(model, model_name):
    '''grads = model.tr
    for p in model.parameters():
        if not p.grad is None:
            grads.append(float(p.grad.mean()))

    grads = np.array(grads)
    if grads.any() and grads.mean() > 100:
        print("WARNING! gradients mean is over 100 (%s)" % (model_name))
    if grads.any() and grads.max() > 100:
        print("WARNING! gradients max is over 100 (%s)" % (model_name))'''

    #TODO
    return

def generator_model():
    model = Sequential(name="generator")
    model.add(Dense(input_dim=100, units=4*4*512))
    model.add(ReLU())
    model.add(Reshape((4, 4, 512)))
    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same", kernel_initializer="glorot_normal"))
    model.add(ReLU())
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same", kernel_initializer="glorot_normal"))
    model.add(ReLU())
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", kernel_initializer="glorot_normal"))
    model.add(ReLU())
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(989, kernel_size=5, strides=2, padding="same", kernel_initializer="glorot_normal"))
    model.add(Softmax(axis=2))
    model.trainable = True
    model.summary()
    return model

def discriminator_model(blockBackwardsDict, minimapValues):
    model = Sequential(name="discriminator")
    model.add(keras.layers.InputLayer(input_shape=(64, 64, 989)))
    #model.add(GaussianNoise(0.05))
    #model.add(AddMinimapValues(blockBackwardsDict, minimapValues))
    model.add(Conv2D(16, kernel_size=3, strides=1, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, kernel_size=3, strides=1, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size=3, strides=1, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same", kernel_initializer="glorot_normal"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    #model.add(MinibatchDiscrimination(nb_kernels=25, kernel_dim=15))
    model.add(Dense(1, activation="sigmoid", kernel_initializer="glorot_normal"))
    model.trainable = True
    model.summary()
    return model

def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model

def train(epochs, batch_size, worldCount, versionName = None):
    curDir = os.getcwd()
    ganDir = utils.check_or_create_local_path("softgan")

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
    blockForwardDict, blockBackwardDict = utils.load_encoding_dict("optimized")

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
        d_optim = SGD(lr=0.0001)
        g_optim = Adam(lr=0.0001, beta_1=0.5)

        d = discriminator_model(blockBackwardDict, minimapValues)
        d.compile(loss="binary_crossentropy", optimizer=d_optim, metrics=["accuracy"])

        g = generator_model()
        d_on_g = generator_containing_discriminator(g, d)

        d_on_g.compile(loss="binary_crossentropy", optimizer=g_optim)

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
    X_train = load_worlds(worldCount, "%s\\WorldRepo2\\" % curDir, 64, 64, blockForwardDict, minimapValues, utilizationCount)

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
            noise = np.random.normal(0, 0.5, size=(batch_size, 100))
            fakeWorlds = g.predict(noise)

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

            #Traing generator on X data, with Y labels
            noise = np.random.normal(0, 0.5, (batch_size, 100))

            #Train generator to generate real
            g_loss = d_on_g.train_on_batch(noise, realLabels)
            gLossSummary.value[0].simple_value = g_loss
            tbWriter.add_summary(gLossSummary, (epoch * numberOfBatches) + minibatchIndex)
            tbWriter.flush()

            print("epoch [%d/%d] :: batch [%d/%d] :: disAcc = %.1f%% :: disLoss = %f :: genLoss = %f" % (epoch, epochs, minibatchIndex, numberOfBatches, d_loss[1] * 100, d_loss[0], g_loss))

            #Save models
            if minibatchIndex % 100 == 99 or minibatchIndex == numberOfBatches - 1:
                d.save("%s\\discriminator.h5" % versionDir)
                g.save("%s\\generator.h5" % versionDir)

            #Debugging
            if d_loss[1] == 1 or d_loss[0] == 0:
                batchesLasted = (epoch * numberOfBatches) + minibatchIndex + 1
                print ("Tune lasted %s" % batchesLasted)
                return

        # Perform some checks on the gradients of each model
        check_grads(d, "discriminator")
        check_grads(g, "generator")



def main():
    #tests.test_addminimap_layer()
    train(epochs=1000, batch_size=32, worldCount=5000)

if __name__ == "__main__":
    main()

