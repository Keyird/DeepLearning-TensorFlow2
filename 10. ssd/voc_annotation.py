import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'test')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# 解析xml,获得标签值，并向txt中写入标签
def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
            
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

# 返回当前进程的工作目录
wd = getcwd()

# 写入图片路径和对应的标签值
for year, image_set in sets:
    # 读取ImageSets/train.txt、test.txt 文件中的每一行（图片的id号）
    image_ids = open('VOCdevkit/VOC%s/ImageSets/%s.txt'%(year, image_set)).read().strip().split()
    # 以只写方式打开文件
    list_file = open('files/%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        # 在相应的文件中写入图片的路径名
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
        # 写入对应的标签
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
