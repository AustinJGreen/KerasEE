import numpy as np
import scipy
import utils
import os
import argparse

from keras.applications import inception_v3
from pro_classifier import build_classifier
from keras.models import load_model
from keras import backend as K

# These are the names of the layers
# for which we try to maximize activation,
# as well as their weight in the final loss
# we try to maximize.
# You can tweak these setting to obtain new visual effects.
scale = 10
rand_params = (np.random.rand(1, 4)[0] * scale) - (scale / 2)
good_params = [-0.5, 1.9, 1.3, 0.5]
interesting_params = [-1.87673595,  2.23979389,  3.22691387, -2.42502887]

params = rand_params
print(params)

settings = {
    'features': {
        'max_pooling2d_1': params[0],
        'max_pooling2d_2': params[1],
        'max_pooling2d_3': params[2],
        'max_pooling2d_4': params[3]
    },
}

K.set_learning_phase(0)

# Build the InceptionV3 network with our placeholder.
# The model will be loaded with pre-trained ImageNet weights.
model = load_model(
    'C:\\Users\\austi\\Documents\\PycharmProjects\\KerasEE\\models\\pro_classifier\\ver38\\models\\latest.h5')

dream = model.input
print('Model loaded.')

# Get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# Define the loss.
loss = K.variable(0.)
for layer_name in settings['features']:
    # Add the L2 norm of the features of a layer to the loss.
    if layer_name not in layer_dict:
        raise ValueError('Layer ' + layer_name + ' not found in model.')
    coeff = settings['features'][layer_name]
    x = layer_dict[layer_name].output
    # We avoid border artifacts by only involving non-border pixels in the loss.
    scaling = K.prod(K.cast(K.shape(x), 'float32'))
    if K.image_data_format() == 'channels_first':
        loss += coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
    else:
        loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

# Compute the gradients of the dream wrt the loss.
grads = K.gradients(loss, dream)[0]
# Normalize gradients.
grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

# Set up function to retrieve the value
# of the loss and gradients given an input image.
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def resize_world(world, size):
    world = np.copy(world)
    factors = (1,
               float(size[0]) / world.shape[1],
               float(size[1]) / world.shape[2],
               1)
    return scipy.ndimage.zoom(world, factors, order=1)


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x = np.add(x, np.multiply(step, grad_values, casting='unsafe'), casting='unsafe')
    return x


"""
Process:
- Load the original image.
- Define a number of processing scales (i.e. image shapes),
    from smallest to largest.
- Resize the original image to the smallest scale.
- For every scale, starting with the smallest (i.e. current one):
    - Run gradient ascent
    - Upscale image to the next scale
    - Reinject the detail that was lost at upscaling time
- Stop when we are back to the original size.
To obtain the detail lost during upscaling, we simply
take the original image, shrink it down, upscale it,
and compare the result to the (resized) original image.
"""

# Playing with these hyperparameters will also allow you to achieve new effects
step = 0.01  # Gradient ascent step size
num_octave = 3  # Number of scales at which to run gradient ascent
octave_scale = 1.25  # Size ratio between scales
iterations = 100  # Number of ascent steps per scale
max_loss = 1

# Load resources
cur_dir = os.getcwd()
res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))

block_images = utils.load_block_images(res_dir)
block_forward, block_backward = utils.load_encoding_dict(res_dir, 'blocks_optimized')

world_id = 'PW7W3RcLd3cEI'

world_data = utils.load_world_data_ver3(f'{res_dir}\\worlds\\{world_id}.world')

# Get center section
world_size = world_data.shape[0]
input_size = int(model.input.shape[1])
start = (world_size // 2) - (input_size // 2)
end = start + input_size
world_input_data = world_data[start:end, start:end]

world_encoded = np.array([utils.encode_world_sigmoid(block_forward, world_input_data)])
original_shape = world_encoded.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
original_world = np.copy(world_encoded)
shrunk_original_world = resize_world(world_encoded, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    world_encoded = resize_world(world_encoded, shape)
    world_encoded = gradient_ascent(world_encoded,
                                    iterations=iterations,
                                    step=step,
                                    max_loss=max_loss)
    upscaled_shrunk_original_world = resize_world(shrunk_original_world, shape)
    same_size_original = resize_world(original_world, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_world
    world_encoded += lost_detail
    shrunk_original_world = resize_world(original_world, shape)

world_dream = utils.decode_world_sigmoid(block_backward, np.copy(world_encoded)[0])
utils.save_world_preview(block_images, world_dream, f'{cur_dir}\\dream.png')
utils.save_world_preview(block_images, world_input_data, f'{cur_dir}\\input.png')
