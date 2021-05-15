from tensorflow.keras import layers, Model, Sequential

# 基础残差模块：实用于层数较少的ResNet-18/ResNet-34
class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        # 第一个卷积层+BN层
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                   padding="SAME", use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # 第二个卷积层+BN层
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                   padding="SAME", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs  # 一般情况直接进行恒等映射
        if self.downsample is not None:  # 如果需要进行下采样，执行下采样函数
            identity = self.downsample(inputs)
        # 残差模块中的第一层
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        # 残差模块中的第二层
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        # 按对应通道相加
        x = self.add([identity, x])
        x = self.relu(x)

        return x

# 第二种残差模块Bottlenect：适用于层数很深的ResNet-50/ResNet-101/ResNet-152
class Bottleneck(layers.Layer):
    expansion = 4

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        # 第一层：采用1*1卷积核进行降维
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False, name="conv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")
        # 第二层：采用3*3卷积层
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                   strides=strides, padding="SAME", name="conv2")
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")
        # 第三层：采用1*1卷积核进行升维（注意原论文中这一层的输出的通道数要为输入的4倍，因此这里使用了倍数expansion=4）
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3")
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")

        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)
        # 第一层
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        # 第三层
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        # 按对应通道相加
        x = self.add([x, identity])
        x = self.relu(x)

        return x

# 将相同的残差模块（比如Bottlenect）连接成一个大的残差块，对于原论文表1中的conv2_x、conv3_x、conv4_x、conv5_x
def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None
    # 当步长为2或者输出维度不等于输入的维度时，需在shortcut上增加1*1卷积层，改变shortcut支路输出的shape或者维度
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, name="conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")
    # 实现表1中的一个大的残差块
    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))
    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)


def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=True):
    # 定义输入(batch, 224, 224, 3)
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    # 第一层conv1
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, name="conv1")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)
    # conv2_x
    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
    # conv3_x
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
    # conv4_x
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
    # conv5_x
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)

    if include_top:
        # 全局平均池化
        x = layers.GlobalAvgPool2D()(x)
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x

    model = Model(inputs=input_image, outputs=predict)

    return model

def resnet34(im_width=224, im_height=224, num_classes=1000):
    return _resnet(BasicBlock, [3, 4, 6, 3], im_width, im_height, num_classes)

def resnet50(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(Bottleneck, [3, 4, 6, 3], im_width, im_height, num_classes, include_top)

def resnet101(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(Bottleneck, [3, 4, 23, 3], im_width, im_height, num_classes, include_top)

