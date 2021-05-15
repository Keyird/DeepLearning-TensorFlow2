"""
function: Get test.txt and trainval.txt
"""
import os
import random 

# 路径
xmlfilepath = 'Annotations'
saveBasePath = "ImageSets"

train_percent = 0.9

# 返回指定路径下的文件列表
temp_xml = os.listdir(xmlfilepath)

# 存放xmlfilepath路径下所有的xml文件
total_xml = []
for xml in temp_xml:
    # endswith() 方法用于判断字符串是否以指定后缀结尾
    if xml.endswith(".xml"):
        total_xml.append(xml)

# 数据集总容量
num = len(total_xml)
train = int(num * train_percent)

list = range(num)
# 用于截取列表的指定长度的随机数，但是不会改变列表本身的排序
train = random.sample(list, train)

ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
 
for i in list:
    name = total_xml[i][:-4]+'\n'  # 取图片前面的序号
    if i in train:
        ftrain.write(name)
    else:  
        ftest.write(name)  
  
ftrain.close()
ftest .close()
