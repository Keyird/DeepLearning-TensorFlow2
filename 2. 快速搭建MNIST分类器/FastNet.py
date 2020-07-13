import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics  # 导入TF子库

# 1.数据集准备
(x, y), (x_val, y_val) = datasets.mnist.load_data()  # 加载数据集，返回的是两个元组，分别表示训练集和测试集
# （1）训练集
x = tf.convert_to_tensor(x, dtype=tf.float32)/255.  # 转换为张量，并缩放到0~1
y = tf.convert_to_tensor(y, dtype=tf.int32)  # 转换为张量（标签）
y = tf.one_hot(y, depth=10)  # 热独编码
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))  # 构建数据集对象
train_dataset = train_dataset.shuffle(60000).batch(128)  # 打乱顺序，设置批量训练的batch为128
# （2）验证集
x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)/255.
y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
y_val = tf.one_hot(y_val, depth=10)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(128)

# 2.网络搭建
network = Sequential([
    layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])
network.build(input_shape=[None, 28*28])
network.summary()

# 3.模型装配
network.compile(optimizer=optimizers.Adam(lr=0.01),  # 指定优化器
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),  # 指定采用交叉熵损失函数，包含Softmax
                metrics=['accuracy'])  # 指定评价指标为准备率

# 4.模型训练
history = network.fit(train_dataset, epochs=20, validation_data=val_dataset, validation_freq=2)
network.evaluate(val_dataset)  # 打印输出loss和accuracy


