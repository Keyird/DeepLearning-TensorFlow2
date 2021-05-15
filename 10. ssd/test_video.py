from detect import SSD
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import time

fps = 0.0
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

ssd = SSD()
# capture=cv2.VideoCapture(0) # 调用摄像头
video = "改成自己的视频路径"
capture = cv2.VideoCapture(video)

while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame = capture.read()
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    frame = np.array(ssd.detect_image(frame))
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    fps = (fps + (1./(time.time()-t1))) / 2
    # print("fps= %.2f" % (fps))
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("video",frame)
    cv2.waitKey(1)

