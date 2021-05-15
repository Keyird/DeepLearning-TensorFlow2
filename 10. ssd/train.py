from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from networks.loss import MultiboxLoss
from networks.generator import Generator
from networks.anchors import get_anchors
from networks.utils import BBoxUtility
from networks.ssd import SSD300
import tensorflow as tf
import numpy as np

# 获取当前主机上所有GPU设备列表
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    # 设置仅在需要时申请显存空间
    tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == "__main__":

    weights_dir = "weights/"  # 模型保存路径
    annotation_path = 'files/2007_train.txt'
    NUM_CLASSES = 21  # 数据集类别：20+1
    input_shape = (300, 300, 3)
    # 获得6个不同预测分支的Anchors大小
    priors = get_anchors()
    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    # 加载预训练模型
    model.load_weights('weights/ssd_pre_weights.h5', by_name=True, skip_mismatch=True)

    # 训练参数设置
    logging = TensorBoard(log_dir=weights_dir)
    # 训练途中会进行保存模型，每隔1个epo进行保存，只保存最好val_loss的epo
    checkpoint = ModelCheckpoint(weights_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    # 每隔两个epo，如果val_loss没有降低，学习率就降低为原来的一半
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    # 当val_loss有6个epo没有下降时，就停止训练
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    # 在训练初期，冻结网络前21层，加快训练速度
    freeze_layer = 21
    for i in range(freeze_layer):
        model.layers[i].trainable = False

    if True:
        # BATCH_SIZE不要太小，不然训练效果很差
        BATCH_SIZE = 8
        # 学习率，粗略地训练
        Lr = 5e-4
        # 为起始世代
        Init_Epoch = 0
        # 为冻结训练的世代
        Freeze_Epoch = 50

        # 数据生成器
        gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                    (input_shape[0], input_shape[1]), NUM_CLASSES)

        # 模型的装配
        model.compile(optimizer=Adam(lr=Lr), loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)

        # 0-50个epoches时，冻结对网络前21层的训练
        model.fit(gen.generate(True),
                steps_per_epoch=num_train//BATCH_SIZE,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=Freeze_Epoch,
                initial_epoch=Init_Epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    # 解冻前21层参数
    for i in range(freeze_layer):
        model.layers[i].trainable = True

    if True:
        # BATCH_SIZE不要太小，不然训练效果很差
        BATCH_SIZE = 8
        # 学习率减小精细训练
        Lr = 1e-4
        Freeze_Epoch = 50
        Epoch = 70

        # 数据生成器
        gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:],
                    (input_shape[0], input_shape[1]),NUM_CLASSES)

        model.compile(optimizer=Adam(lr=Lr),loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=3.0).compute_loss)

        # 训练地50-70个epoches
        model.fit(gen.generate(True),
                steps_per_epoch=num_train//BATCH_SIZE,
                validation_data=gen.generate(False),
                validation_steps=num_val//BATCH_SIZE,
                epochs=Epoch,
                initial_epoch=Freeze_Epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])