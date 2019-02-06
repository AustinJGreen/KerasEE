from keras import backend as K
from keras.layers import Input, Conv2D, UpSampling2D, LeakyReLU, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam

from src.pconv_layer import PConv2D


class PConvUnet:

    def __init__(self, feature_model, feature_layers, inference_only=False):
        """Create the PConvUnet."""

        self.inference_only = inference_only

        # Set current epoch
        self.current_epoch = 0

        # Layers to extract features from (first maxpooling layers)
        self.feature_layers = feature_layers

        self.features = self.build_feature_model(feature_model)

        # Create UNet-like model
        self.model = self.build_pconv_unet()

    def build_feature_model(self, feature_model):
        world = Input(shape=(128, 128, 10))

        # If inference only, just return empty model
        if self.inference_only:
            model = Model(inputs=world, outputs=[world for _ in range(len(self.feature_layers))])
            model.trainable = False
            model.compile(loss='mse', optimizer='adam')
            return model

        # Output the first three pooling layers
        feature = feature_model
        feature.outputs = [feature_model.layers[i].output for i in self.feature_layers]

        # Create model and compile
        model = Model(inputs=world, outputs=feature(world))
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')

        return model

    def build_pconv_unet(self, train_bn=True, lr=0.0002):

        # INPUTS
        inputs_world = Input((128, 128, 10), name='inputs_world')
        inputs_mask = Input((128, 128, 10), name='inputs_mask')

        # ENCODER
        def encoder_layer(img_in, mask_in, filters, kernel_size, bn=True):
            conv1, mask1 = PConv2D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])
            if bn:
                conv1 = BatchNormalization(momentum=0.8)(conv1, training=train_bn)
            conv1 = Activation('relu')(conv1)

            conv2, mask2 = PConv2D(filters, kernel_size, strides=1, padding='same')([conv1, mask1])
            if bn:
                conv2 = BatchNormalization(momentum=0.8)(conv2, training=train_bn)
            conv2 = Activation('relu')(conv2)

            return conv2, mask2

        e_conv1, e_mask1 = encoder_layer(inputs_world, inputs_mask, 64, 7, bn=False)
        e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5)
        e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5)
        e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 512, 3)
        e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 512, 3)

        # DECODER
        def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):
            up_img = UpSampling2D(size=(2, 2))(img_in)
            up_mask = UpSampling2D(size=(2, 2))(mask_in)
            concat_img = Concatenate(axis=3)([e_conv, up_img])
            concat_mask = Concatenate(axis=3)([e_mask, up_mask])
            conv1, mask1 = PConv2D(filters, kernel_size, padding='same')([concat_img, concat_mask])
            if bn:
                conv1 = BatchNormalization(momentum=0.8)(conv1)
            conv1 = LeakyReLU(alpha=0.2)(conv1)
            conv2, mask2 = PConv2D(filters, kernel_size, padding='same')([conv1, mask1])
            if bn:
                conv2 = BatchNormalization(momentum=0.8)(conv2)
            conv2 = LeakyReLU(alpha=0.2)(conv2)
            return conv2, mask2

        d_conv9, d_mask9 = decoder_layer(e_conv5, e_mask5, e_conv4, e_mask4, 512, 3)
        d_conv10, d_mask10 = decoder_layer(d_conv9, d_mask9, e_conv3, e_mask3, 512, 3)
        d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv2, e_mask2, 256, 5)
        d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, e_conv1, e_mask1, 128, 5)
        d_conv16, d_mask16 = decoder_layer(d_conv12, d_mask12, inputs_world, inputs_mask, 64, 7, bn=False)
        outputs = Conv2D(10, 1, activation='sigmoid')(d_conv16)

        # Setup the model inputs / outputs
        model = Model(inputs=[inputs_world, inputs_mask], outputs=outputs)

        # Compile the model
        model.compile(
            optimizer=Adam(lr=lr),
            loss=self.loss_total(inputs_mask)
        )

        return model

    def loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components 
        and multiplies by their weights. See paper eq. 7.
        """

        def loss(y_true, y_pred):
            # Compute predicted image with non-hole pixels set to ground truth
            y_comp = mask * y_true + (1 - mask) * y_pred

            # Compute the features
            dm_out = self.features(y_pred)
            dm_gt = self.features(y_true)
            dm_comp = self.features(y_comp)

            # Compute loss components
            l1 = self.loss_valid(mask, y_true, y_pred)
            l2 = self.loss_hole(mask, y_true, y_pred)
            l3 = self.loss_perceptual(dm_out, dm_gt, dm_comp)
            l4 = self.loss_style(dm_out, dm_gt)
            l5 = self.loss_style(dm_comp, dm_gt)
            l6 = 0  # self.loss_tv(mask, y_comp)

            # Return loss function
            return l1 + 6 * l2 + 0.05 * l3 + 120 * (l4 + l5) + 0.1 * l6

        return loss

    def loss_hole(self, mask, y_true, y_pred):
        """Pixel L1 loss within the hole / mask"""
        return self.l1((1 - mask) * y_true, (1 - mask) * y_pred)

    def loss_valid(self, mask, y_true, y_pred):
        """Pixel L1 loss outside the hole / mask"""
        return self.l1(mask * y_true, mask * y_pred)

    def loss_perceptual(self, vgg_out, vgg_gt, vgg_comp):
        """Perceptual loss based on VGG16, see. eq. 3 in paper"""
        loss = 0
        for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
            loss += self.l1(o, g) + self.l1(c, g)
        return loss

    def loss_style(self, output, vgg_gt):
        """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
        loss = 0
        for o, g in zip(output, vgg_gt):
            loss += self.l1(self.gram_matrix(o), self.gram_matrix(g))
        return loss

    def loss_tv(self, mask, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = K.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
        dilated_mask = K.conv2d(1 - mask, kernel, data_format='channels_last', padding='same')

        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
        p = dilated_mask * y_comp

        # Calculate total variation loss
        a = self.l1(p[:, 1:, :, :], p[:, :-1, :, :])
        b = self.l1(p[:, :, 1:, :], p[:, :, :-1, :])
        return a + b

    def fit(self, generator, epochs=10, plot_callback=None, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator

        param generator: training generator yielding (maskes_image, original_image) tuples
        param epochs: number of epochs to train for
        param plot_callback: callback function taking Unet model as parameter
        """

        # Loop over epochs
        for epoch in range(epochs):

            # Fit the model
            self.model.fit_generator(
                generator,
                epochs=self.current_epoch + 1,
                initial_epoch=self.current_epoch,
                *args, **kwargs
            )

            # Update epoch 
            self.current_epoch += 1

            # After each epoch predict on test images & show them
            if plot_callback:
                plot_callback(self.model)

    @staticmethod
    def l1(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.sum(K.abs(y_pred - y_true), axis=[1, 2, 3])
        elif K.ndim(y_true) == 3:
            return K.sum(K.abs(y_pred - y_true), axis=[1, 2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")

    @staticmethod
    def gram_matrix(x):
        """Calculate gram matrix used in style loss"""

        # Assertions on input
        assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
        assert K.image_data_format() == 'channels_last', "Please use channels-last format"

        # Permute channels and get resulting shape
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        b, c, h, w = shape[0], shape[1], shape[2], shape[3]

        # Reshape x and do batch dot product
        features = K.reshape(x, K.stack([b, c, h * w]))
        gram = K.batch_dot(features, features, axes=2)

        # Normalize with channels, height and width
        gram = gram / K.cast(c * h * w, x.dtype)

        return gram
