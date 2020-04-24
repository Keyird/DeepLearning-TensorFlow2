from model import resnet50
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf

im_height = 224
im_width = 224

# 加载要进行预测的图片
img = Image.open("E:/DeepLearning/ResNet-101/rose_test.jpg")
# resize成224x224
img = img.resize((im_width, im_height))
plt.imshow(img)

# 将图片做预处理
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
img = np.array(img).astype(np.float32)
img = img - [_R_MEAN, _G_MEAN, _B_MEAN]

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))

# class_indices.json中存放的是标签字典
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)
# 网络模型的微调
feature = resnet50(num_classes=5, include_top=False)
feature.trainable = False
model = tf.keras.Sequential([feature,
                             tf.keras.layers.GlobalAvgPool2D(),
                             tf.keras.layers.Dropout(rate=0.5),
                             tf.keras.layers.Dense(1024),
                             tf.keras.layers.Dropout(rate=0.5),
                             tf.keras.layers.Dense(5),
                             tf.keras.layers.Softmax()])
# model.build((None, 224, 224, 3))  # when using subclass model
# 加载训练好的模型参数
model.load_weights('./save_weights/resNet_101.ckpt')
result = model.predict(img)
prediction = np.squeeze(result)
predict_class = np.argmax(result)
print('预测该图片类别是：', class_indict[str(predict_class)], ' 预测概率是：', prediction[predict_class])
plt.show()
