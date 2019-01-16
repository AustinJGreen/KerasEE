import math
import time
from multiprocessing import Process

import numpy as np

import utils


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

    def update_estimate(self, time_points, time0, time1, cnt0, cnt1):
        cnt_delta = cnt1 - cnt0
        worlds_left = self.targetCount - self.worldCounter.value
        if cnt_delta != 0:
            time_points[self.timePtIndex] = (time1 - time0) / cnt_delta
            self.timePtIndex = (self.timePtIndex + 1) % len(time_points)
            if self.timePtIndex < len(time_points):
                self.timePtCnt += 1

        if self.timePtCnt > 0:
            time_left_sec = np.average(time_points[0:self.timePtCnt]) * worlds_left
            time_left = time_left_sec / 60.0
            time_left_minutes = int(time_left)
            time_left_minutes_frac = time_left - time_left_minutes
            time_left_seconds = math.ceil(time_left_minutes_frac * 60)
            if time_left_minutes > 0:
                return "ETA %i minutes" % time_left_minutes
            else:
                return "ETA %i seconds" % time_left_seconds  # "ETA <1 Minute"
        else:
            return ""

    @staticmethod
    def is_good_world(cross_section):
        # Count blocks?
        # Count action blocks?
        edited_blocks = 0
        width = cross_section.shape[0]
        height = cross_section.shape[1]
        for x in range(width):
            for y in range(height):
                block = cross_section[x, y]
                if block != 0:
                    edited_blocks += 1

        total_size = width * height
        return edited_blocks >= 0.4 * total_size

    def load_world(self, world_file):
        world = utils.load_world_data_ver3(world_file)
        world_width = world.shape[0]
        world_height = world.shape[1]

        if world_width < self.genWidth or world_height < self.genHeight:
            return

        x_margin = world_width % self.genWidth
        y_margin = world_height % self.genHeight

        x_offset = 0
        y_offset = 0

        x_min_increment = 0.5 * self.genWidth
        y_min_increment = 0.5 * self.genHeight

        if x_margin > 0:
            x_offset = np.random.randint(0, x_margin)

        if y_margin > 0:
            y_offset = np.random.randint(0, y_margin)

        x_start = x_offset
        while x_start + self.genWidth < world_width:

            y_start = y_offset
            while y_start + self.genHeight < world_height:
                x_end = x_start + self.genWidth
                y_end = y_start + self.genHeight
                cross_section = world[x_start:x_end, y_start:y_end]

                cross_section0 = cross_section

                if self.is_good_world(cross_section0):

                    encoded_world0 = utils.encode_world2d_binary(cross_section0)

                    encoded_worlds = [encoded_world0]

                    self.threadLock.acquire()

                    local_index = 0
                    while self.worldCounter.value < self.targetCount and local_index < len(encoded_worlds):
                        self.loadQueue.put(encoded_worlds[local_index])
                        self.worldCounter.value += 1
                        local_index += 1

                    self.threadLock.release()

                    if local_index == 0:
                        break

                y_start += np.random.randint(y_min_increment, self.genHeight)

            x_start += np.random.randint(x_min_increment, self.genWidth)

    def __init__(self, file_queue, thread_manager, world_counter, thread_lock, target_count, gen_width, gen_height,
                 block_forward_dict, minimap_values):
        Process.__init__(self)
        self.fileQueue = file_queue
        self.loadQueue = thread_manager.Queue()
        self.worldCounter = world_counter
        self.threadLock = thread_lock
        self.targetCount = int(target_count)
        self.genWidth = gen_width
        self.genHeight = gen_height
        self.blockForwardDict = block_forward_dict
        self.minimapValues = minimap_values

        self.daemon = True

    def run(self):
        time_points = np.array([0.] * 250)
        while not self.fileQueue.empty() and self.worldCounter.value < self.targetCount:
            world_file = self.fileQueue.get()
            time0 = time.time()
            cnt0 = self.worldCounter.value
            self.load_world(world_file)
            time1 = time.time()
            cnt1 = self.worldCounter.value
            time_est_str = self.update_estimate(time_points, time0, time1, cnt0, cnt1)
            print("Loaded (%s/%s) %s" % (self.worldCounter.value, self.targetCount, time_est_str))
            if self.worldCounter.value >= self.targetCount:
                break
        print("Done loading.")

    def get_worlds(self):
        return self.loadQueue
