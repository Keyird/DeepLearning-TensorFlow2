import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

class BasicBlock(layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()  # BN层
        self.relu = layers.Activation('relu')  # ReLU激活函数

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()  # BN层

        if stride != 1:
            self.downsample = Sequential()  # 下采样
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x  # 恒等映射

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out,training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out,training=training)

        identity = self.downsample(inputs)  # 恒等映射

        output = layers.add([out, identity])  # 主路与支路（恒等映射）相加
        output = tf.nn.relu(output)  # ReLU激活函数

        return output

class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=100):
        super(ResNet, self).__init__()
        # 第一层
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
        # 中间层的四个残差块：conv2_x，conv3_x，conv4_x，conv5_x
        self.layer1 = self.build_resblock(64,  layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        # 全局平均池化
        self.avgpool = layers.GlobalAveragePooling2D()
        # 全连接层
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs,training=training)
        x = self.layer1(x,training=training)
        x = self.layer2(x,training=training)
        x = self.layer3(x,training=training)
        x = self.layer4(x,training=training)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    # 构建残差块（将几个相同的残差模块堆叠在一起）
    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        # 可能会进行下采样
        res_blocks.add(BasicBlock(filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks

def resnet18():
    return ResNet([2, 2, 2, 2])

def resnet34():
    return ResNet([3, 4, 6, 3])
