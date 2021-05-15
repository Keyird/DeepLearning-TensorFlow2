&emsp;&emsp;熟悉我的博友都知道，最近我在写一个[《TF2.0深度学习实战：图像分类/目标检测》](https://blog.csdn.net/wjinjie/article/details/104700834)的小白教程。但是就在上一期实现[ResNet](https://blog.csdn.net/wjinjie/article/details/105583526)的过程中，由于电脑性能原因，我不得不选择层数较少的ResNet-18进行训练。但是很快我发现，虽然只有18层，传统的训练方法仍然很耗时，甚至难以完成对101层的ResNet-101的训练。  
&emsp;&emsp;出于这个原因，这一次，我将采用一种巧妙的方法——迁移学习来实现。即在预训练模型的基础上，采用很深的深度残差网络ResNet-101，对如下图所示的花数据集进行训练，快速实现对原始图像的分类和预测，最终预测精确度达到了惊人的98%。

代码已经上传仓库，具体实现过程和详细讲解，可参考我的博客：[震惊！！掌握这个技巧，轻松训练好一个100层的超深神经网络！](https://blog.csdn.net/wjinjie/article/details/105665214) 

预训练模型下载链接：[ResNet-101预训练模型](https://pan.baidu.com/s/1KYgGBH_MCvMQ3oGxBpIeHw)， 提取码：dg2m   
数据集下载链接：[花分类数据集](https://pan.baidu.com/s/1YLjWX0z09brMewe-kKbWZg)， 提取码：9ao5
