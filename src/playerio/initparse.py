import numpy as np


def parse(m):
    if m is None:
        raise Exception('Message cannot be null')

    if m.type != 'init' and m.type != 'reset':
        raise Exception('Invalid message type')

    p = 0
    data = []

    while m[p] != 'ws':
        p += 1

    p += 1

    while m[p] != 'we':
        data.append(m[p])
        p += 1

    chunks = []
    while len(data) > 0:
        args = []
        while len(data) > 0 and not isinstance(data[-1], bytes):
            args.insert(0, data.pop())

        ys = list(memoryview(data.pop()))
        xs = list(memoryview(data.pop()))
        layer = data.pop()
        block_type = data.pop()

        chunks.append(DataChunk(layer, block_type, xs, ys, args))

    return chunks


def get_world_data(m):
    world_width = m[18]
    world_height = m[19]

    blocks = np.empty((world_width, world_height, 2), dtype=BlockData)

    for i in range(world_width):
        for j in range(world_height):
            for k in range(2):
                block_id = 0
                if i == 0 or i == world_width - 1 or j == 0 or j == world_height - 1:
                    block_id = 9

                blocks[i, j, k] = BlockData(k, i, j, block_id, None)

    data = parse(m)
    for chunk in data:
        for location in chunk.locations:
            blocks[location[0], location[1], chunk.layer] = BlockData(chunk.layer, location[0], location[1],
                                                                      int(chunk.type), chunk.args)
    return blocks


class BlockData:
    layer = None
    x = None
    y = None
    block_id = None
    args = None

    def __init__(self, layer, x, y, block_id, args):
        self.layer = layer
        self.x = x
        self.y = y
        self.block_id = block_id
        self.args = args


class DataChunk:
    locations = None
    args = None
    layer = None
    type = None

    def __init__(self, layer, block_type, xs, ys, args):
        self.layer = layer
        self.type = block_type
        self.args = args
        self.locations = self.get_locations(xs, ys)

    @staticmethod
    def get_locations(xs, ys):
        points = []
        for i in range(0, len(xs), 2):
            points.append((((xs[i] << 8) | xs[i + 1]), ((ys[i] << 8) | ys[i + 1])))
        return points
