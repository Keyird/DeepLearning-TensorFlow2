import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

def VGG16(input_tensor):

    # 新建一个空字典，存放网络结构
    net = {}
    # input：300x300x3
    net['input'] = input_tensor

    # Block1：300x300x3 -> 150x150x64
    net['conv1_1'] = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', name='conv1_1')(net['input'])
    net['conv1_2'] = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', name='conv1_2')(net['conv1_1'])
    net['maxpool1'] = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='maxpool1')(net["conv1_2"])

    # Block2：150x150x64 -> 75x75x128
    net['conv2_1'] = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', name='conv2_1')(net['maxpool1'])
    net['conv2_2'] = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same', name='conv2_2')(net['conv2_1'])
    net['maxpool2'] = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name='maxpool2')(net["conv2_2"])

    # Block3：75x75x128 -> 38x38x256
    net['conv3_1'] = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3_1')(net['maxpool2'])
    net['conv3_2'] = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3_3')(net['conv3_2'])
    net['maxpool3'] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool3')(net["conv3_2"])

    # Block4：38x38x256 -> 19x19x512
    net['conv4_1'] = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv4_1')(net['maxpool3'])
    net['conv4_2'] = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv4_3')(net['conv4_2'])
    net['maxpool4'] = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='maxpool4')(net['conv4_3'])

    # Block5：19x19x512 -> 19x19x512（待改进）
    net['conv5_1'] = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5_1')(net['maxpool4'])
    net['conv5_2'] = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5_3')(net['conv5_2'])
    net['maxpool5'] = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', name='maxpool5')(net['conv5_3'])

    # Conv6(FC6)：19x19x512 -> 19x19x1024 空洞卷积
    net['fc6'] = Conv2D(1024, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(net['maxpool5'])

    # Conv7(FC7): 19x19x1024 -> 19x19x1024
    net['fc7'] = Conv2D(1024, kernel_size=(1, 1), activation='relu', padding='same', name='fc7')(net['fc6'])

    # Conv8: 19x19x1024 -> 10x10x512
    net['conv8_1'] = Conv2D(256, kernel_size=(1, 1), activation='relu', padding='same', name='conv8_1')(net['fc7'])
    net['conv8_2'] = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv8_padding')(net['conv8_1'])
    net['conv8_2'] = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', name='conv8_2')(net['conv8_2'])

    # Conv9: 10x10x512 -> 5x5x256
    net['conv9_1'] = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same', name='conv9_1')(net['conv8_2'])
    net['conv9_2'] = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv9_padding')(net['conv9_1'])
    net['conv9_2'] = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv9_2')(net['conv9_2'])

    # Conv10: 5x5x256 -> 3x3x256  无填充
    net['conv10_1'] = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same', name='conv10_1')(net['conv9_2'])
    net['conv10_2'] = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid', name='conv10_2')(net['conv10_1'])

    # Conv11: 3x3x256 -> 1x1x256
    net['conv11_1'] = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same', name='conv11_1')(net['conv10_2'])
    net['conv11_2'] = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='valid', name='conv11_2')(net['conv11_1'])

    return net