"""
author: AI JUN
function: VGG11 by python
date: 2020/4/2
"""
import tensorflow as tf  # 导入TF库
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics  # 导入TF子库
import os, glob
import random, csv
import matplotlib.pyplot as plt

# 创建图片路径和标签，并写入csv文件
def load_csv(root, filename, name2label):
    # root:数据集根目录
    # filename:csv文件名
    # name2label:类别名编码表
    if not os.path.exists(os.path.join(root, filename)):  # 如果不存在csv，则创建一个
        images = []  # 初始化存放图片路径的字符串数组
        for name in name2label.keys():  # 遍历所有子目录，获得所有图片的路径
            # glob文件名匹配模式，不用遍历整个目录判断而获得文件夹下所有同类文件
            # 只考虑后缀为png,jpg,jpeg的图片，比如：pokemon\\mewtwo\\00001.png
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))
        print(len(images), images)  # 打印出images的长度和所有图片路径名
        random.shuffle(images)  # 随机打乱存放顺序
        # 创建csv文件，并且写入图片路径和标签信息
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:  # 遍历images中存放的每一个图片的路径，如pokemon\\mewtwo\\00001.png
                name = img.split(os.sep)[-2]  # 用\\分隔，取倒数第二项作为类名
                label = name2label[name]  # 找到类名键对应的值，作为标签
                writer.writerow([img, label])  # 写入csv文件，以逗号隔开，如：pokemon\\mewtwo\\00001.png, 2
            print('written into csv file:', filename)
    # 读csv文件
    images, labels = [], []  # 创建两个空数组，用来存放图片路径和标签
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:  # 逐行遍历csv文件
            img, label = row  # 每行信息包括图片路径和标签
            label = int(label)  # 强制类型转换为整型
            images.append(img)  # 插入到images数组的后面
            labels.append(label)
    assert len(images) == len(labels)  # 断言，判断images和labels的长度是否相同
    return images, labels

# 首先遍历pokemon根目录下的所有子目录。对每个子目录，用类别名作为编码表的key，编码表的长度作为类别的标签，存进name2label字典对象
def load_pokemon(root, mode='train'):
    # 创建数字编码表
    name2label = {}  # 创建一个空字典{key:value}，用来存放类别名和对应的标签
    for name in sorted(os.listdir(os.path.join(root))):  # 遍历根目录下的子目录，并排序
        if not os.path.isdir(os.path.join(root, name)):  # 如果不是文件夹，则跳过
            continue
        name2label[name] = len(name2label.keys())   # 给每个类别编码一个数字
    images, labels = load_csv(root, 'images.csv', name2label)  # 读取csv文件中已经写好的图片路径，和对应的标签
    # 将数据集按6：2：2的比例分成训练集、验证集、测试集
    if mode == 'train':  # 60%
        images = images[:int(0.6 * len(images))]
        labels = labels[:int(0.6 * len(labels))]
    elif mode == 'val':  # 20% = 60%->80%
        images = images[int(0.6 * len(images)):int(0.8 * len(images))]
        labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]
    else:  # 20% = 80%->100%
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]
    return images, labels, name2label

img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])

def normalize(x, mean=img_mean, std=img_std):
    x = (x - mean)/std
    return x

# def denormalize(x, mean=img_mean, std=img_std):
    # x = x * std + mean
    # return x

def preprocess(image_path, label):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(image_path)  # 读入图片
    x = tf.image.decode_jpeg(x, channels=3)  # 将原图解码为通道数为3的三维矩阵
    x = tf.image.resize(x, [244, 244])
    # 数据增强
    # x = tf.image.random_flip_up_down(x) # 上下翻转
    # x = tf.image.random_flip_left_right(x)  # 左右镜像
    x = tf.image.random_crop(x, [224, 224, 3])  # 裁剪
    x = tf.cast(x, dtype=tf.float32) / 255.  # 归一化
    x = normalize(x)
    y = tf.convert_to_tensor(label)  # 转换为张量
    return x, y

# 1.加载自定义数据集
images, labels, table = load_pokemon('pokemon', 'train')
print('images', len(images), images)
print('labels', len(labels), labels)
print(table)
db = tf.data.Dataset.from_tensor_slices((images, labels))  # images: string path， labels: number
db = db.shuffle(1000).map(preprocess).batch(16).repeat(20)

# 2.网络搭建
network = Sequential([
    # 第一层
    layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 第二层
    layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 第三层
    layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第四层
    layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 第五层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第六层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    # 第七层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    # 第八层
    layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.Flatten(),  # 拉直 7*7*512
    # 第九层
    layers.Dense(1024, activation='relu'),
    layers.Dropout(rate=0.5),
    # 第十层
    layers.Dense(128, activation='relu'),
    layers.Dropout(rate=0.5),
    # 第十一层
    layers.Dense(5, activation='softmax')
])
network.build(input_shape=(None, 224, 224, 3))  # 设置输入格式
network.summary()  # 打印各层参数表

# 3.模型训练（计算梯度，迭代更新网络参数）
optimizer = optimizers.SGD(lr=0.01)  # 声明采用批量随机梯度下降方法，学习率=0.01
acc_meter = metrics.Accuracy()
x_step = []
y_accuracy = []
for step, (x, y) in enumerate(db):  # 一次输入batch组数据进行训练
    with tf.GradientTape() as tape:  # 构建梯度记录环境
        x = tf.reshape(x, (-1, 224, 224, 3))  # 将输入拉直，[b,28,28]->[b,784]
        out = network(x)  # 输出[b, 10]
        y_onehot = tf.one_hot(y, depth=5)  # one-hot编码
        loss = tf.square(out - y_onehot)
        loss = tf.reduce_sum(loss)/16  # 定义均方差损失函数，注意此处的32对应为batch的大小
        grads = tape.gradient(loss, network.trainable_variables)  # 计算网络中各个参数的梯度
        optimizer.apply_gradients(zip(grads, network.trainable_variables))  # 更新网络参数
        acc_meter.update_state(tf.argmax(out, axis=1), y)  # 比较预测值与标签，并计算精确度
    if step % 10 == 0:  # 每200个step，打印一次结果
        print('Step', step, ': Loss is: ', float(loss), ' Accuracy: ', acc_meter.result().numpy())
        x_step.append(step)
        y_accuracy.append(acc_meter.result().numpy())
        acc_meter.reset_states()

# 4.可视化
plt.plot(x_step, y_accuracy, label="training")
plt.xlabel("step")
plt.ylabel("accuracy")
plt.title("accuracy of training")
plt.legend()
plt.show()



