import numpy as np
import utils
import time
import math
from multiprocessing import Process
from multiprocessing import Queue

class GanWorldLoader(Process):
    fileQueue = None
    loadQueue = None
    worldCounter = None
    targetCount = 0

    genWidth = 0
    genHeight = 0
    blockForwardDict = None
    minimapValues = None

    threadLock = None

    timePtIndex = 0
    timePtCnt = 0

    def update_estimate(self, timePoints, time0, time1, cnt0, cnt1):
        cntDelta = cnt1 - cnt0
        worldsLeft = self.targetCount - self.worldCounter.value
        if cntDelta != 0:
            timePoints[self.timePtIndex] = (time1 - time0) / cntDelta
            self.timePtIndex = (self.timePtIndex + 1) % len(timePoints)
            if self.timePtIndex < len(timePoints):
                    self.timePtCnt += 1

        if self.timePtCnt > 0:
            timeLeftSec = np.average(timePoints[0:self.timePtCnt]) * worldsLeft
            timeLeft = timeLeftSec / 60.0
            timeLeftMinutes = int(timeLeft)
            timeLeftMinutesFrac = timeLeft - timeLeftMinutes
            timeLeftSeconds = math.ceil(timeLeftMinutesFrac * 60)
            if timeLeftMinutes > 0:
                return "ETA %i minutes" % (timeLeftMinutes)
            else:
                return "ETA %i seconds" % (timeLeftSeconds) #"ETA <1 Minute"
        else:
            return ""

    def load_world(self, worldFile):
        world = utils.load_world_data_ver3(worldFile)
        worldWidth = world.shape[0]
        worldHeight = world.shape[1]

        xMargin = worldWidth % self.genWidth
        yMargin = worldHeight % self.genHeight

        horizontalCrossSections = int((worldWidth - xMargin) / self.genWidth)
        verticalCrossSections = int((worldHeight - yMargin) / self.genHeight)

        xOffset = 0
        yOffset = 0

        if xMargin > 0:
            xOffset = np.random.randint(0, xMargin)
            yOffset = np.random.randint(0, yMargin)

        for yCrossSection in range(verticalCrossSections):
            for xCrossSection in range(horizontalCrossSections):
                xStart = xOffset + (xCrossSection * self.genWidth)
                xEnd = xStart + self.genWidth
                yStart = yOffset + (yCrossSection * self.genHeight)
                yEnd = yStart + self.genHeight
                crossSection = world[xStart:xEnd, yStart:yEnd]

                blockCnt = 0
                uniqueCnt = 0
                uniqueBlocks = {}

                for y in range(self.genHeight):
                    for x in range(self.genWidth):
                        id = crossSection[x, y]
                        if id not in uniqueBlocks:
                            uniqueBlocks[id] = 1
                            uniqueCnt += 1
                        else:
                            uniqueBlocks[id] += 1

                        if id != 0:
                            blockCnt += 1

                if blockCnt >= (0.5 * (self.genWidth * self.genHeight)) and uniqueCnt >= 10:
                    crossSection0 = crossSection
                    crossSection90 = utils.rotate_world90(crossSection0)
                    crossSection180 = utils.rotate_world90(crossSection90)
                    crossSection270 = utils.rotate_world90(crossSection180)

                    encodedWorld0 = utils.encode_world2d(self.blockForwardDict, crossSection0)
                    encodedWorld90 = utils.encode_world2d(self.blockForwardDict, crossSection90)
                    encodedWorld180 = utils.encode_world2d(self.blockForwardDict, crossSection180)
                    encodedWorld270 = utils.encode_world2d(self.blockForwardDict, crossSection270)

                    encodedWorlds = [ encodedWorld0, encodedWorld90, encodedWorld180, encodedWorld270 ]

                    self.threadLock.acquire()

                    localIndex = 0
                    while self.worldCounter.value < self.targetCount and localIndex < 4:
                        self.loadQueue.put(encodedWorlds[localIndex])
                        self.worldCounter.value += 1
                        localIndex += 1

                    self.threadLock.release()

                    if localIndex == 0:
                        break

    def __init__(self, fileQueue, threadManager, worldCounter, threadLock, targetCount, genWidth, genHeight, blockForwardDict, minimapValues):
        Process.__init__(self)
        self.fileQueue = fileQueue
        self.loadQueue = threadManager.Queue()
        self.worldCounter = worldCounter
        self.threadLock = threadLock
        self.targetCount = int(targetCount)
        self.genWidth = genWidth
        self.genHeight = genHeight
        self.blockForwardDict = blockForwardDict
        self.minimapValues = minimapValues

        self.daemon = True

    def run(self):
        timePoints = np.array([0.] * 250)
        while not self.fileQueue.empty() and self.worldCounter.value < self.targetCount:
            worldFile = self.fileQueue.get()
            time0 = time.time()
            cnt0 = self.worldCounter.value
            self.load_world(worldFile)
            time1 = time.time()
            cnt1 = self.worldCounter.value
            timeEstStr = self.update_estimate(timePoints, time0, time1, cnt0, cnt1)
            print ("Loaded (%s/%s) %s" % (self.worldCounter.value, self.targetCount, timeEstStr))
            if self.worldCounter.value >= self.targetCount:
                break

    def get_worlds(self):
        return self.loadQueue

class SoftGanWorldLoader(Process):
    fileQueue = None
    loadQueue = None
    worldCounter = None
    targetCount = 0

    genWidth = 0
    genHeight = 0
    blockForwardDict = None
    minimapValues = None

    threadLock = None

    timePtIndex = 0
    timePtCnt = 0

    def update_estimate(self, timePoints, time0, time1, cnt0, cnt1):
        cntDelta = cnt1 - cnt0
        worldsLeft = self.targetCount - self.worldCounter.value
        if cntDelta != 0:
            timePoints[self.timePtIndex] = (time1 - time0) / cntDelta
            self.timePtIndex = (self.timePtIndex + 1) % len(timePoints)
            if self.timePtIndex < len(timePoints):
                self.timePtCnt += 1

        if self.timePtCnt > 0:
            timeLeftSec = np.average(timePoints[0:self.timePtCnt]) * worldsLeft
            timeLeft = timeLeftSec / 60.0
            timeLeftMinutes = int(timeLeft)
            timeLeftMinutesFrac = timeLeft - timeLeftMinutes
            timeLeftSeconds = math.ceil(timeLeftMinutesFrac * 60)
            if timeLeftMinutes > 0:
                return "ETA %i minutes" % (timeLeftMinutes)
            else:
                return "ETA %i seconds" % (timeLeftSeconds)  # "ETA <1 Minute"
        else:
            return ""

    def load_world(self, worldFile):
        world = utils.load_world_data(worldFile)
        worldWidth = world.shape[0]
        worldHeight = world.shape[1]

        xMargin = worldWidth % self.genWidth
        yMargin = worldHeight % self.genHeight

        horizontalCrossSections = int((worldWidth - xMargin) / self.genWidth)
        verticalCrossSections = int((worldHeight - yMargin) / self.genHeight)

        xOffset = 0
        yOffset = 0

        if xMargin > 0:
            xOffset = np.random.randint(0, xMargin)
            yOffset = np.random.randint(0, yMargin)

        for yCrossSection in range(verticalCrossSections):
            for xCrossSection in range(horizontalCrossSections):
                xStart = xOffset + (xCrossSection * self.genWidth)
                xEnd = xStart + self.genWidth
                yStart = yOffset + (yCrossSection * self.genHeight)
                yEnd = yStart + self.genHeight
                crossSection = world[xStart:xEnd, yStart:yEnd]

                blockCnt = 0
                enoughBlocks = False
                for y in range(self.genHeight):
                    for x in range(self.genWidth):
                        if crossSection[x, y, 0] != 0:
                            blockCnt += 1
                            if blockCnt >= ((self.genWidth * self.genHeight) * 0.33):
                                enoughBlocks = True
                                break
                    if enoughBlocks:
                        break

                if enoughBlocks:
                    encodedWorld = utils.encode_world_softmax(self.blockForwardDict, crossSection)

                    self.threadLock.acquire()
                    addWorld = self.worldCounter.value < self.targetCount
                    if addWorld:
                        self.loadQueue.put(encodedWorld)
                        self.worldCounter.value += 1
                    self.threadLock.release()

                    if not addWorld:
                        break

    def __init__(self, fileQueue, threadManager, worldCounter, threadLock, targetCount, genWidth, genHeight,
                 blockForwardDict, minimapValues):
        Process.__init__(self)
        self.fileQueue = fileQueue
        self.loadQueue = threadManager.Queue()
        self.worldCounter = worldCounter
        self.threadLock = threadLock
        self.targetCount = int(targetCount)
        self.genWidth = genWidth
        self.genHeight = genHeight
        self.blockForwardDict = blockForwardDict
        self.minimapValues = minimapValues

        self.daemon = True

    def run(self):
        timePoints = np.array([0.] * 250)
        while not self.fileQueue.empty() and self.worldCounter.value < self.targetCount:
            worldFile = self.fileQueue.get()
            time0 = time.time()
            cnt0 = self.worldCounter.value
            self.load_world(worldFile)
            time1 = time.time()
            cnt1 = self.worldCounter.value
            timeEstStr = self.update_estimate(timePoints, time0, time1, cnt0, cnt1)
            print("Loaded (%s/%s) %s" % (self.worldCounter.value, self.targetCount, timeEstStr))
            if self.worldCounter.value >= self.targetCount:
                break

    def get_worlds(self):
        return self.loadQueue
