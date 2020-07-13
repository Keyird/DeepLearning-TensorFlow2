from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import GoogLeNet
import tensorflow as tf
import json
import os

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # 根路径
# 此处要根据自己文件存放路径来改
image_path = data_root + "DeepLearning/GoogLeNet/flower_data/"  # flower数据集路径
train_dir = image_path + "train"  # 训练集路径
validation_dir = image_path + "val"  # 验证集路径

# 创建文件save_weights用来存放训练好的模型
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")

im_height = 224
im_width = 224
batch_size = 32
epochs = 30

def pre_function(img):
    # img = im.open('test.jpg')
    # img = np.array(img).astype(np.float32)
    img = img / 255.    # 归一化
    img = (img - 0.5) * 2.0   # 标准化
    return img

# 定义训练集图像生成器，并对图像进行预处理
train_image_generator = ImageDataGenerator(preprocessing_function=pre_function,
                                           horizontal_flip=True)  # 水平翻转
# 定义验证集图像生成器，并对图像进行预处理
validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)
# 使用图像生成器从文件夹train_dir中读取样本，默认对标签进行了one-hot编码
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')  # 分类方式
total_train = train_data_gen.n  # 训练集样本数
class_indices = train_data_gen.class_indices  # 数字编码标签字典：{类别名称：索引}
inverse_dict = dict((val, key) for key, val in class_indices.items())  # 转换字典中键与值的位置
json_str = json.dumps(inverse_dict, indent=4)  # 将转换后的字典写入文件class_indices.json
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)
# 使用图像生成器从验证集validation_dir中读取样本
val_data_gen = train_image_generator.flow_from_directory(directory=validation_dir,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         target_size=(im_height, im_width),
                                                         class_mode='categorical')
total_val = val_data_gen.n  # 验证集样本数
model = GoogLeNet(im_height=im_height, im_width=im_width, class_num=5, aux_logits=True)  # 实例化模型
# model.build((batch_size, 224, 224, 3))  # when using subclass model
model.summary()  # 每层参数信息

# 使用keras底层api进行网络训练。
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)  # 定义损失函数（这种方式需要one-hot编码）
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)  # 优化器

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')  # 定义平均准确率

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        aux1, aux2, output = model(images, training=True)
        loss1 = loss_object(labels, aux1)   # 辅助分类器损失函数
        loss2 = loss_object(labels, aux2)
        loss3 = loss_object(labels, output)  # 主分类器损失函数
        loss = loss1*0.3 + loss2*0.3 + loss3  # 总损失函数
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, output)


@tf.function
def test_step(images, labels):
    _, _, output = model(images, training=False)
    t_loss = loss_object(labels, output)
    test_loss(t_loss)
    test_accuracy(labels, output)

best_test_loss = float('inf')
for epoch in range(1, epochs+1):
    train_loss.reset_states()        # 训练损失值清零
    train_accuracy.reset_states()    # clear history info
    test_loss.reset_states()         # clear history info
    test_accuracy.reset_states()     # clear history info

    for step in range(total_train // batch_size):
        images, labels = next(train_data_gen)
        train_step(images, labels)

    for step in range(total_val // batch_size):
        test_images, test_labels = next(val_data_gen)
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    if test_loss.result() < best_test_loss:
        best_test_loss = test_loss.result()
        model.save_weights("./save_weights/myGoogLeNet.h5")   # 保存模型为.h5格式
