import numpy as np
import scipy
import utils
import os

from keras.models import load_model
from keras import backend as K

# These are the names of the layers
# for which we try to maximize activation,
# as well as their weight in the final loss
# we try to maximize.
# You can tweak these setting to obtain new visual effects.
scale = 10
rand_params = (np.random.rand(1, 4)[0] * scale) - (scale / 2)

subtle = [1.03186447,  4.94900884,  4.52247586, -1.00151375]
colors = [2.84376834, -4.96601078, -0.03940069, -4.7203666]

params = colors
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
cur_dir = os.getcwd()
kerasee_dir = os.path.abspath(os.path.join(cur_dir, '..'))
res_dir = os.path.abspath(os.path.join(cur_dir, '..', 'res'))
models_dir = os.path.abspath(os.path.join(cur_dir, '..', 'models'))
model = load_model(f'{models_dir}\\pro_classifier\\ver38\\models\\latest.h5')

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
    loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

# Compute the gradients of the dream wrt the loss and normalize gradients
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

# Set up function to retrieve the value
# of the loss and gradients given an input image.
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(input_x):
    outs = fetch_loss_and_grads([input_x])
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


def gradient_ascent(grad_x, ascent_cnt, step_value, maximum_loss=None):
    for ascent_i in range(ascent_cnt):
        loss_value, grad_values = eval_loss_and_grads(grad_x)
        if maximum_loss is not None and loss_value > maximum_loss:
            break
        grad_x = np.add(grad_x, step_value * grad_values, casting='unsafe')
    return grad_x


def dream(world_data):
    # Playing with these hyperparameters will also allow you to achieve new effects
    step = 0.05  # Gradient ascent step size
    num_octave = 4  # Number of scales at which to run gradient ascent
    octave_scale = 1.1  # Size ratio between scales
    iterations = 100  # Number of ascent steps per scale
    max_loss = 3

    # Load resources
    block_images = utils.load_block_images(res_dir)
    block_forward, block_backward = utils.load_encoding_dict(res_dir, 'blocks_optimized')

    world_encoded = np.array([utils.encode_world_sigmoid(block_forward, world_data)])
    world_orig = np.copy(world_encoded)

    original_shape = world_encoded.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octave):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]

    for shape in successive_shapes:
        print('Processing image shape', shape)
        world_encoded = resize_world(world_encoded, shape)
        world_encoded = gradient_ascent(world_encoded,
                                        ascent_cnt=iterations,
                                        step_value=step,
                                        maximum_loss=max_loss)

    world_dream = utils.decode_world_sigmoid(block_backward, world_encoded[0])
    utils.save_world_preview(block_images, world_dream, f'{kerasee_dir}\\dream.png', overwrite=True)
    utils.save_world_preview(block_images, world_data, f'{kerasee_dir}\\input.png', overwrite=True)


def main():
    # world_data = utils.load_world_live('PWcxcdMHDna0I', credential='kw')
    world_data = utils.load_world_eelvl('C:\\Users\\austi\\Desktop\\PWlzL4SX1ecUI.eelvl')
    # world_id = 'PWrO5qmOGjb0I'
    # world_data = utils.load_world_data_ver3(f'{res_dir}\\worlds\\{world_id}.world')
    dream(world_data)


if __name__ == '__main__':
    main()
