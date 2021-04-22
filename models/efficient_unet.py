import collections
import warnings

import tensorflow as tf


# default parameters ----------------------------------------------------------
DEFAULT_BACKBONE = "efficientnetb0"
DEFAULT_ACTIVATION = "relu"
DEFAULT_DECODER_BLOCK_TYPE = "upsampling"

backbones = collections.namedtuple('backbone', ['name', 'model', 'layers'])

Backbones = [
    backbones(
        "efficientnetb0",
        tf.keras.applications.EfficientNetB0,
        ('block6a_expand_activation', 'block4a_expand_activation',
         'block3a_expand_activation', 'block2a_expand_activation')),
    
    backbones(
        "efficientnetb1",
        tf.keras.applications.EfficientNetB1,
        ('block6a_expand_activation', 'block4a_expand_activation',
         'block3a_expand_activation', 'block2a_expand_activation')),
    
    backbones(
        "efficientnetb2",
        tf.keras.applications.EfficientNetB2,
        ('block6a_expand_activation', 'block4a_expand_activation',
         'block3a_expand_activation', 'block2a_expand_activation')),
    
    backbones(
        "efficientnetb3",
        tf.keras.applications.EfficientNetB3,
        ('block6a_expand_activation', 'block4a_expand_activation',
         'block3a_expand_activation', 'block2a_expand_activation')),
    
    backbones(
        "efficientnetb4",
        tf.keras.applications.EfficientNetB4,
        ('block6a_expand_activation', 'block4a_expand_activation',
         'block3a_expand_activation', 'block2a_expand_activation')),
    
    backbones(
        "efficientnetb5",
        tf.keras.applications.EfficientNetB5,
        ('block6a_expand_activation', 'block4a_expand_activation',
         'block3a_expand_activation', 'block2a_expand_activation')),
    backbones(
        "efficientnetb6",
        tf.keras.applications.EfficientNetB6,
        ('block6a_expand_activation', 'block4a_expand_activation',
         'block3a_expand_activation', 'block2a_expand_activation')),
    
    backbones(
        "efficientnetb7",
        tf.keras.applications.EfficientNetB7,
        ('block6a_expand_activation', 'block4a_expand_activation',
         'block3a_expand_activation', 'block2a_expand_activation')),
]


# helper functions ------------------------------------------------------------
def conv_block(
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_batchnorm=False,
        dropout_rate=0.25,
        **kwargs
):
    """Extension of Conv2D layer with batchnorm"""
    activation_functions = {
        "relu": tf.nn.relu,
        "leaky_relu": tf.nn.leaky_relu
    }

    conv_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop('name', None)
    
    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name + '_bn'

    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    if activation is not None:
        if str(activation).lower() not in activation_functions.keys():
            warnings.warn(f"activation must be in {list(activation_functions.keys())}, received: {activation}. Setting decoder_block_type='{DEFAULT_ACTIVATION}'.")
            activation = activation_functions[DEFAULT_ACTIVATION]
   
        else:
            activation = activation_functions[str(activation).lower()]

    def wrapper(input_tensor):

        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if activation:
            x = tf.keras.layers.Activation(activation, name=act_name)(x)
            
        if dropout_rate is not None:
            x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
            
        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        return x

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = f'decoder_stage{stage}_upsampling'
    conv1_name = f'decoder_stage{stage}a'
    conv2_name = f'decoder_stage{stage}b'
    concat_name = f'decoder_stage{stage}_concat'

    concat_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):
        x = tf.keras.layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = tf.keras.layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = conv_block(
            filters,
            kernel_size=3,
            activation='leaky_relu',
            use_batchnorm=use_batchnorm,
            padding='same',
            kernel_initializer='he_uniform',
            name=conv1_name)(x)

        x = conv_block(
            filters,
            kernel_size=3,
            activation='leaky_relu',
            use_batchnorm=use_batchnorm,
            padding='same',
            kernel_initializer='he_uniform',
            name=conv2_name)(x)


        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = f'decoder_stage{stage}a_transpose'
    bn_name = f'decoder_stage{stage}a_bn'
    relu_name = f'decoder_stage{stage}a_relu'
    conv_block_name = f'decoder_stage{stage}b'
    concat_name = f'decoder_stage{stage}_concat'

    concat_axis = bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):

        x = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        x = tf.keras.layers.LeakyReLU(name=relu_name)(x)
        
        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        if skip is not None:
            x = tf.keras.layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        # x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)
        x = conv_block(
            filters,
            kernel_size=3,
            activation='leaky_relu',
            use_batchnorm=use_batchnorm,
            padding='same',
            kernel_initializer='he_uniform',
            name=conv_block_name)(x)

    return wrapper


def efficient_Unet(
    encoder_weights='imagenet',
    decoder_block_type="upsampling",
    backbone="efficientnetb3",
    num_classes=4,
    activation="softmax",
    input_shape=(None, None, 3),
    decoder_filters=(256, 128, 64, 32, 16),
    use_batchnorm=True
):
    decoder_blocks = {
        "upsampling": DecoderUpsamplingX2Block,
        "transpose": DecoderTransposeX2Block
    }

    # set decoder block type
    if decoder_block_type in decoder_blocks.keys():
        decoder_block = decoder_blocks[decoder_block_type]

    else:
        warnings.warn(f"decoder_block_type must be in {list(decoder_blocks.keys())}, received: {decoder_block_type}. Setting decoder_block_type='{DEFAULT_DECODER_BLOCK_TYPE}'.")
        decoder_block = decoder_blocks[DEFAULT_DECODER_BLOCK_TYPE]

    # set backbone model
    backbone_names = [i.name for i in Backbones]

    if backbone not in backbone_names:
        warnings.warn(f"backbone must be in {backbone_names}, received: {backbone}. setting backbone_name='{DEFAULT_BACKBONE}'.")
        backbone = backbone_names.index(DEFAULT_BACKBONE)
    
    else:
        backbone = backbone_names.index(backbone)

    backbone_model = Backbones[backbone].model(
        include_top=False,
        weights=encoder_weights,
        input_shape=input_shape,
        pooling=None,
        classes=num_classes,
        classifier_activation='softmax')

    skip_connection_layers = Backbones[backbone].layers

    # build model
    x = backbone_model.output

    # output of skipped connection layers
    skips = ([backbone_model.get_layer(name=i).output if isinstance(i, str)
            else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # building decoder blocks
    for i in range(len(decoder_filters)):
        if i < len(skips):
            skip = skips[i]
        
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    # final layers
    x = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)

    x = tf.keras.layers.Activation(activation, name=activation)(x)

    # create keras model instance
    return tf.keras.models.Model(inputs=backbone_model.input, outputs=x)


if __name__ is "__main__":
    model = efficient_Unet(
        decoder_block_type="upsampling",
        backbone="efficientnetb3",
        num_classes=4,
        activation="softmax",
        input_shape=(512, 512, 3))

    model.summary()