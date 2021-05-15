import tensorflow as tf
from tensorflow.keras import layers

# 瓶颈层，相当于每一个稠密块中若干个相同的H函数
class BottleNeck(layers.Layer):
    # growth_rate对应的是论文中的增长率k，指经过一个BottleNet输出的特征图的通道数；drop_rate指失活率。
    def __init__(self, growth_rate, drop_rate):
        super(BottleNeck, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(filters=4 * growth_rate,  # 使用1*1卷积核将通道数降维到4*k
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters=growth_rate,  # 使用3*3卷积核，使得输出维度（通道数）为k
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.dropout = layers.Dropout(rate=drop_rate)
        # 将网络层存入一个列表中
        self.listLayers = [self.bn1,
                           layers.Activation("relu"),
                           self.conv1,
                           self.bn2,
                           layers.Activation("relu"),
                           self.conv2,
                           self.dropout]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        # 每经过一个BottleNet，将输入和输出按通道连结。作用是：将前l层的输入连结起来，作为下一个BottleNet的输入。
        y = layers.concatenate([x, y], axis=-1)
        return y

# 稠密块，由若干个相同的瓶颈层构成
class DenseBlock(layers.Layer):
    # num_layers表示该稠密块存在BottleNet的个数，也就是一个稠密块的层数L
    def __init__(self, num_layers, growth_rate, drop_rate=0.5):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.listLayers = []
        # 一个DenseBlock由多个相同的BottleNeck构成，我们将它们放入一个列表中。
        for _ in range(num_layers):
            self.listLayers.append(BottleNeck(growth_rate=self.growth_rate, drop_rate=self.drop_rate))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x

# 过渡层
class TransitionLayer(layers.Layer):
    # out_channels代表输出通道数
    def __init__(self, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = layers.BatchNormalization()
        self.conv = layers.Conv2D(filters=out_channels,
                                           kernel_size=(1, 1),
                                           strides=1,
                                           padding="same")
        self.pool = layers.MaxPool2D(pool_size=(2, 2),   # 2倍下采样
                                              strides=2,
                                              padding="same")

    def call(self, inputs):
        x = self.bn(inputs)
        x = tf.keras.activations.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

# DenseNet整体网络结构
class DenseNet(tf.keras.Model):
    # num_init_features:代表初始的通道数，即输入稠密块时的通道数
    # growth_rate:对应的是论文中的增长率k，指经过一个BottleNet输出的特征图的通道数
    # block_layers:每个稠密块中的BottleNet的个数
    # compression_rate:压缩因子，其值在(0,1]范围内
    # drop_rate：失活率
    def __init__(self, num_init_features, growth_rate, block_layers, compression_rate, drop_rate):
        super(DenseNet, self).__init__()
        # 第一层，7*7的卷积层，2倍下采样。
        self.conv = layers.Conv2D(filters=num_init_features,
                                           kernel_size=(7, 7),
                                           strides=2,
                                           padding="same")
        self.bn = layers.BatchNormalization()
        # 最大池化层，3*3卷积核，2倍下采样
        self.pool = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")

        # 稠密块 Dense Block(1)
        self.num_channels = num_init_features
        self.dense_block_1 = DenseBlock(num_layers=block_layers[0], growth_rate=growth_rate, drop_rate=drop_rate)
        # 该稠密块总的输出的通道数
        self.num_channels += growth_rate * block_layers[0]
        # 对特征图的通道数进行压缩
        self.num_channels = compression_rate * self.num_channels
        # 过渡层1，过渡层进行下采样
        self.transition_1 = TransitionLayer(out_channels=int(self.num_channels))

        # 稠密块 Dense Block(2)
        self.dense_block_2 = DenseBlock(num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[1]
        self.num_channels = compression_rate * self.num_channels
        # 过渡层2，2倍下采样，输出：14*14
        self.transition_2 = TransitionLayer(out_channels=int(self.num_channels))

        # 稠密块 Dense Block(3)
        self.dense_block_3 = DenseBlock(num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels
        # 过渡层3，2倍下采样
        self.transition_3 = TransitionLayer(out_channels=int(self.num_channels))

        # 稠密块 Dense Block(4)
        self.dense_block_4 = DenseBlock(num_layers=block_layers[3], growth_rate=growth_rate, drop_rate=drop_rate)

        # 全局平均池化，输出size：1*1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 全连接层，进行10分类
        self.fc = layers.Dense(units=10, activation=tf.keras.activations.softmax)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = tf.keras.activations.relu(x)
        x = self.pool(x)

        x = self.dense_block_1(x)
        x = self.transition_1(x)
        x = self.dense_block_2(x)
        x = self.transition_2(x)
        x = self.dense_block_3(x)
        x = self.transition_3(x,)
        x = self.dense_block_4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

def densenet():
    return DenseNet(num_init_features=64, growth_rate=32, block_layers=[2,2,2,2], compression_rate=0.5, drop_rate=0.5)
    # return DenseNet(num_init_features=64, growth_rate=32, block_layers=[4, 4, 4, 4], compression_rate=0.5, drop_rate=0.5)
mynet=densenet()




