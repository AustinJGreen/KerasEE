import math
import os
import random
import time
from multiprocessing import Process, Manager, Value, Lock

import numpy as np

import utils


def load_worlds(load_count, world_directory, gen_width, gen_height, minimap_values, block_forward, thread_count):
    world_names = os.listdir(world_directory)
    random.shuffle(world_names)

    with Manager() as manager:
        file_queue = manager.Queue()

        for name in world_names:
            file_queue.put(world_directory + name)

        world_array = np.zeros((load_count, gen_width, gen_height, 10), dtype=np.int8)

        world_counter = Value('i', 0)
        thread_lock = Lock()

        threads = [None] * thread_count
        for thread in range(thread_count):
            load_thread = WorldLoader(file_queue, manager, world_counter, thread_lock, load_count, gen_width,
                                      gen_height, block_forward, minimap_values)
            load_thread.start()
            threads[thread] = load_thread

        world_index = 0
        for thread in range(thread_count):
            threads[thread].join()
            print("Thread %s joined." % thread)
            thread_load_queue = threads[thread].get_worlds()
            print("Adding worlds to list from thread %s queue." % thread)
            while thread_load_queue.qsize() > 0:
                world_array[world_index] = thread_load_queue.get()
                world_index += 1
            print("Done adding worlds to list from thread.")

        world_array = world_array[:world_index, :, :, :]
    return world_array


class WorldLoader(Process):
    file_queue = None
    load_queue = None
    world_counter = None
    target_count = 0

    gen_width = 0
    gen_height = 0
    block_forward = None
    minimap_values = None

    thread_lock = None

    time_pt_index = 0
    time_pt_cnt = 0

    def update_estimate(self, time_points, time0, time1, cnt0, cnt1):
        cnt_delta = cnt1 - cnt0
        worlds_left = self.target_count - self.world_counter.value
        if cnt_delta != 0:
            time_points[self.time_pt_index] = (time1 - time0) / cnt_delta
            self.time_pt_index = (self.time_pt_index + 1) % len(time_points)
            if self.time_pt_index < len(time_points):
                self.time_pt_cnt += 1

        if self.time_pt_cnt > 0:
            time_left_sec = np.average(time_points[0:self.time_pt_cnt]) * worlds_left
            time_left = time_left_sec / 60.0
            time_left_minutes = int(time_left)
            time_left_minutes_frac = time_left - time_left_minutes
            time_left_seconds = math.ceil(time_left_minutes_frac * 60)
            if time_left_minutes > 1:
                return "ETA %i minutes" % time_left_minutes
            elif time_left_minutes == 1:
                return "ETA 1 minute"
            else:
                return "ETA %i seconds" % time_left_seconds  # "ETA <1 Minute"
        else:
            return ""

    @staticmethod
    def is_good_world(cross_section):
        # - Count blocks
        # - Diversity of blocks
        edited_blocks = 0
        distinct_ids = []
        width = cross_section.shape[0]
        height = cross_section.shape[1]
        for x in range(width):
            for y in range(height):
                block = cross_section[x, y]
                if block != 0:
                    edited_blocks += 1
                if block not in distinct_ids:
                    distinct_ids.append(block)

        total_size = width * height
        required = int(0.4 * total_size)
        return edited_blocks >= required and len(distinct_ids) >= 5

    def load_world(self, world_file):
        world = utils.load_world_data_ver3(world_file)
        world_width = world.shape[0]
        world_height = world.shape[1]

        if world_width < self.gen_width or world_height < self.gen_height:
            return

        x_margin = world_width % self.gen_width
        y_margin = world_height % self.gen_height

        x_offset = 0
        y_offset = 0

        x_min_increment = 1 * self.gen_width
        y_min_increment = 1 * self.gen_height

        if x_margin > 0:
            x_offset = np.random.randint(0, x_margin)

        if y_margin > 0:
            y_offset = np.random.randint(0, y_margin)

        x_start = x_offset
        while x_start + self.gen_width < world_width:

            y_start = y_offset
            while y_start + self.gen_height < world_height:
                x_end = x_start + self.gen_width
                y_end = y_start + self.gen_height
                cross_section = world[x_start:x_end, y_start:y_end]

                cross_section0 = cross_section

                if self.is_good_world(cross_section0):

                    encoded_world0 = utils.encode_world2d_binary(self.block_forward, cross_section0, 10)

                    encoded_worlds = [encoded_world0]

                    self.thread_lock.acquire()

                    local_index = 0
                    while self.world_counter.value < self.target_count and local_index < len(encoded_worlds):
                        self.load_queue.put(encoded_worlds[local_index])
                        self.world_counter.value += 1
                        local_index += 1

                    self.thread_lock.release()

                    if local_index == 0:
                        break

                y_start += np.random.randint(y_min_increment, self.gen_height + 1)

            x_start += np.random.randint(x_min_increment, self.gen_width + 1)

    def __init__(self, file_queue, thread_manager, world_counter, thread_lock, target_count, gen_width, gen_height,
                 block_forward_dict, minimap_values):
        Process.__init__(self)
        self.file_queue = file_queue
        self.load_queue = thread_manager.Queue()
        self.world_counter = world_counter
        self.thread_lock = thread_lock
        self.target_count = int(target_count)
        self.gen_width = gen_width
        self.gen_height = gen_height
        self.block_forward = block_forward_dict
        self.minimap_values = minimap_values

        self.daemon = True

    def run(self):
        time_points = np.array([0.] * 250)
        while not self.file_queue.empty() and self.world_counter.value < self.target_count:
            world_file = self.file_queue.get()
            time0 = time.time()
            cnt0 = self.world_counter.value
            self.load_world(world_file)
            time1 = time.time()
            cnt1 = self.world_counter.value
            time_est_str = self.update_estimate(time_points, time0, time1, cnt0, cnt1)
            print("Loaded (%s/%s) %s" % (self.world_counter.value, self.target_count, time_est_str))
            if self.world_counter.value >= self.target_count:
                break
        print("Done loading.")

    def get_worlds(self):
        return self.load_queue
