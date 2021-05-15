import cv2
from detect import SSD
from PIL import Image
import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

ssd = SSD()
test_img = "D:/Project/SSD/ssd-tf2-myself/img/dog.jpg"

image = cv2.imread(test_img, 1)
image = Image.fromarray(np.uint8(image))

image = np.array(ssd.detect_image(image))
cv2.imshow("object_detect", image)
c = cv2.waitKey(0)



