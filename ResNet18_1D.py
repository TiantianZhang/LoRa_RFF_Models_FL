from keras.layers import *
from keras.models import Model

def identity_block(input_tensor, filter_num, stride, block):

    x = Conv1D(filter_num, 3, strides=stride, padding='same', name=block + 'conv1')(input_tensor)
    x = BatchNormalization(name=block + 'bn1')(x)
    x = Activation(activation='relu', name=block + 'act1')(x)

    x = Conv1D(filter_num, 3, strides=1, padding='same', name=block + 'conv2')(x)
    x = BatchNormalization(name=block + 'bn2')(x)

    if stride != 1:
        x_shortcut = Conv1D(filter_num, 1, strides=stride, name=block + 'shortcut')(input_tensor)
    else:
        x_shortcut = input_tensor

    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x


def resnet18(input_shape, output_shape, model_name='RESNET18'):

    x_input = Input(shape=input_shape)
    min_size = 64
    if input_shape[0] < min_size:
        min_padding = (min_size - input_shape[0]) // 2
        x = ZeroPadding1D((min_padding, min_padding))(x_input)
    else:
        x = x_input

    # Block1
    x = Conv1D(64, 7, 2, padding='same', name='Block1_conv')(x)
    x = BatchNormalization(name='Block1_bn')(x)
    x = Activation('relu', name='Block1_act')(x)
    x = MaxPooling1D(3, 2, name='Block1_pool')(x)

    # Block2
    x = identity_block(x, filter_num=64, stride=1, block='Block2_1_')
    x = identity_block(x, filter_num=64, stride=1, block='Block2_2_')

    # Block3
    x = identity_block(x, filter_num=128, stride=2, block='Block3_1_')
    x = identity_block(x, filter_num=128, stride=1, block='Block3_2_')

    # Block4
    x = identity_block(x, filter_num=256, stride=2, block='Block4_1_')
    x = identity_block(x, filter_num=256, stride=1, block='Block4_2_')

    # Block5
    x = identity_block(x, filter_num=512, stride=2, block='Block5_1_')
    x = identity_block(x, filter_num=512, stride=1, block='Block5_2_')

    # Flatten
    x = Flatten()(x)

    # Dense
    x = Dense(units=output_shape, name='Dense_dense')(x)
    x = BatchNormalization(name='Dense_bn')(x)
    x = Activation(activation='softmax', name='Dense_act')(x)

    # Create model.
    model = Model(x_input, x, name=model_name)

    return model


def resnet18_truncated(input_shape, output_shape, model_name='RESNET18_truncated'):

    x_input = Input(shape=input_shape)
    min_size = 64
    if input_shape[0] < min_size:
        min_padding = (min_size - input_shape[0]) // 2
        x = ZeroPadding1D((min_padding, min_padding))(x_input)
    else:
        x = x_input

    # Block1
    x = Conv1D(64, 7, 2, padding='same', name='Block1_conv')(x)
    x = BatchNormalization(name='Block1_bn')(x)
    x = Activation('relu', name='Block1_act')(x)
    x = MaxPooling1D(3, 2, name='Block1_pool')(x)

    # Block2
    x = identity_block(x, filter_num=64, stride=1, block='Block2_1_')
    x = identity_block(x, filter_num=64, stride=1, block='Block2_2_')

    # Block3
    x = identity_block(x, filter_num=128, stride=2, block='Block3_1_')
    x = identity_block(x, filter_num=128, stride=1, block='Block3_2_')

    # Block4
    x = identity_block(x, filter_num=256, stride=2, block='Block4_1_')
    x = identity_block(x, filter_num=256, stride=1, block='Block4_2_')

    # Block5
    x = identity_block(x, filter_num=512, stride=2, block='Block5_1_')
    x = identity_block(x, filter_num=512, stride=1, block='Block5_2_')

    # Flatten
    x = Flatten()(x)

    # Dense
    x = Dense(units=output_shape, name='Dense_dense')(x)

    # Create model.
    model = Model(x_input, x, name=model_name)

    return model
