from tensorflow.keras.layers import Conv2D, Input, Flatten, Concatenate, Reshape, Activation
from tensorflow.keras.models import Model
from networks.vgg16 import VGG16
from networks.ssd_layers import Normalize
from networks.ssd_layers import PriorBox


def SSD300(input_shape, num_classes=21):

    # 输入300,300,3
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])

    # 定义SSD300网络结构，先装载VGG16
    net = VGG16(input_tensor)

    # 预测分支1
    # 对conv4_3进行处理 38,38,512
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])
    # 每个网格点中先验框的数量
    num_priors = 4
    # 位置预测：38,38,512->38,38,16, 4是指代（x,y,h,w）
    net['conv4_3_norm_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])
    # 分类预测：38,38,512->38,38,84
    net['conv4_3_norm_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='conv4_3_norm_mbox_conf')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf_flat'] = Flatten(name='conv4_3_norm_mbox_conf_flat')(net['conv4_3_norm_mbox_conf'])
    # 获取预测分支Anchors对应的张量
    priorbox = PriorBox(img_size, 30.0, max_size=60.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])

    # 预测分支2
    # 对fc7层进行处理
    num_priors = 6
    # 位置预测：19,19,1024->19,19,24
    net['fc7_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='fc7_mbox_loc')(net['fc7'])
    net['fc7_mbox_loc_flat'] = Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])
    # 分类预测：19,19,1024->19,19,6x21
    net['fc7_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='fc7_mbox_conf')(net['fc7'])
    net['fc7_mbox_conf_flat'] = Flatten(name='fc7_mbox_conf_flat')(net['fc7_mbox_conf'])
    # 获取预测分支Anchors对应的张量
    priorbox = PriorBox(img_size, 60.0, max_size=111.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox(net['fc7'])

    # 预测分支3
    # 对conv8_2进行处理
    num_priors = 6
    # 位置预测：10x10x512->10,10,24
    net['conv8_2_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv8_2_mbox_loc')(net['conv8_2'])
    net['conv8_2_mbox_loc_flat'] = Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])
    # 分类预测：10x10x512->10,10,6x21
    net['conv8_2_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='conv8_2_mbox_conf')(net['conv8_2'])
    net['conv8_2_mbox_conf_flat'] = Flatten(name='conv8_2_mbox_conf_flat')(net['conv8_2_mbox_conf'])
    # 获取预测分支Anchors对应的张量
    priorbox = PriorBox(img_size, 111.0, max_size=162.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox(net['conv8_2'])

    # 预测分支4
    # 对conv9_2进行处理
    num_priors = 6
    # 位置预测：5x5x256->5,5,24
    net['conv9_2_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv9_2_mbox_loc')(net['conv9_2'])
    net['conv9_2_mbox_loc_flat'] = Flatten(name='conv9_2_mbox_loc_flat')(net['conv9_2_mbox_loc'])
    # 分类预测：5x5x256->10,10,6x21
    net['conv9_2_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='conv9_2_mbox_conf')(net['conv9_2'])
    net['conv9_2_mbox_conf_flat'] = Flatten(name='conv9_2_mbox_conf_flat')(net['conv9_2_mbox_conf'])
    # 获取预测分支Anchors对应的张量
    priorbox = PriorBox(img_size, 162.0, max_size=213.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv9_2_mbox_priorbox')
    net['conv9_2_mbox_priorbox'] = priorbox(net['conv9_2'])

    # 预测分支5
    # 对conv10_2进行处理
    num_priors = 4
    # 位置预测：3x3x256 -> 3x3x16
    net['conv10_2_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv10_2_mbox_loc')(net['conv10_2'])
    net['conv10_2_mbox_loc_flat'] = Flatten(name='conv10_2_mbox_loc_flat')(net['conv10_2_mbox_loc'])
    # 分类预测：3x3x256 -> 3,3,4x21
    net['conv10_2_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='conv10_2_mbox_conf')(net['conv10_2'])
    net['conv10_2_mbox_conf_flat'] = Flatten(name='conv10_2_mbox_conf_flat')(net['conv10_2_mbox_conf'])
    # 获取预测分支Anchors对应的张量
    priorbox = PriorBox(img_size, 213.0, max_size=264.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv10_2_mbox_priorbox')
    net['conv10_2_mbox_priorbox'] = priorbox(net['conv10_2'])

    # 预测分支6
    # 对conv11_2进行处理
    num_priors = 4
    # 位置预测：1x1x256 -> 1x1x16
    net['conv11_2_mbox_loc'] =  Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv11_2_mbox_loc')(net['conv11_2'])
    net['conv11_2_mbox_loc_flat'] = Flatten(name='conv11_2_mbox_loc_flat')(net['conv11_2_mbox_loc'])
    # 分类预测：1,1,256 -> 1,1,4x21
    net['conv11_2_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='conv11_2_mbox_conf')(net['conv11_2'])
    net['conv11_2_mbox_conf_flat'] = Flatten(name='conv11_2_mbox_conf_flat')(net['conv11_2_mbox_conf'])
    # 获取预测分支Anchors对应的张量
    priorbox = PriorBox(img_size, 264.0, max_size=315.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv11_2_mbox_priorbox')
    net['conv11_2_mbox_priorbox'] = priorbox(net['conv11_2'])

    # 将6个分支的预测结果进行堆叠
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([net['conv4_3_norm_mbox_loc_flat'],
                                                            net['fc7_mbox_loc_flat'],
                                                            net['conv8_2_mbox_loc_flat'],
                                                            net['conv9_2_mbox_loc_flat'],
                                                            net['conv10_2_mbox_loc_flat'],
                                                            net['conv11_2_mbox_loc_flat']])

    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf')([net['conv4_3_norm_mbox_conf_flat'],
                                                              net['fc7_mbox_conf_flat'],
                                                              net['conv8_2_mbox_conf_flat'],
                                                              net['conv9_2_mbox_conf_flat'],
                                                              net['conv10_2_mbox_conf_flat'],
                                                              net['conv11_2_mbox_conf_flat']])

    net['mbox_priorbox'] = Concatenate(axis=1, name='mbox_priorbox')([net['conv4_3_norm_mbox_priorbox'],
                                                                      net['fc7_mbox_priorbox'],
                                                                      net['conv8_2_mbox_priorbox'],
                                                                      net['conv9_2_mbox_priorbox'],
                                                                      net['conv10_2_mbox_priorbox'],
                                                                      net['conv11_2_mbox_priorbox']])

    # 38x38x16+19x19x24+10x10x24+5x5x24+3x3x16+1x1x16=34928
    # 34928 -> 8732,4
    net['mbox_loc'] = Reshape((-1, 4), name='mbox_loc_final')(net['mbox_loc'])
    # 8732,21
    net['mbox_conf'] = Reshape((-1, num_classes), name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])

    net['predictions'] = Concatenate(axis=2, name='predictions')([net['mbox_loc'],
                                                                  net['mbox_conf'],
                                                                  net['mbox_priorbox']])
    model = Model(net['input'], net['predictions'])

    return model


