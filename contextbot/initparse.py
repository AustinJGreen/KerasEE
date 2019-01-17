import numpy as np


def parse(m):
    if m is None:
        raise Exception("Message cannot be null")

    if m.type != "init" and m.type != "reset":
        raise Exception("Invalid message type")

    p = 0
    data = []

    while m[p] != "ws":
        p += 1

    p += 1

    while m[p] != "we":
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
        for location in chunk.Locations:
            blocks[location[0], location[1], chunk.Layer] = BlockData(chunk.Layer, location[0], location[1],
                                                                      int(chunk.Type), chunk.Args)
    return blocks


class BlockData:
    Layer = None
    X = None
    Y = None
    Id = None
    Args = None

    def __init__(self, layer, x, y, block_id, args):
        self.Layer = layer
        self.X = x
        self.Y = y
        self.Id = block_id
        self.Args = args


class DataChunk:
    Locations = None
    Args = None
    Layer = None
    Type = None

    def __init__(self, layer, block_type, xs, ys, args):
        self.Layer = layer
        self.Type = block_type
        self.Args = args
        self.Locations = self.get_locations(xs, ys)

    @staticmethod
    def get_locations(xs, ys):
        points = []
        for i in range(0, len(xs), 2):
            points.append((((xs[i] << 8) | xs[i + 1]), ((ys[i] << 8) | ys[i + 1])))
        return points
