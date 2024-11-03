'''
分类分析（classification）————病例自动判断分析

1.数据及分析对象
'bc_data.csv'，数据内容来自“威斯康星乳腺癌数据库（Wisconsin Breast Cancer Database）” ，该数据集主要记录了569个病例的32个属性。主要属性/字段如下：
（1）ID：病例的ID；
（2）Diagnosis（诊断结果）：M 为恶性，B 为良性。该数据集共包含357个良性病例和212 个恶性病例；
（3）细胞核的10个特征值，包括radius（半径）、texture（纹理）、perimeter（周长）、面积（area）、平滑度（smoothness）、紧凑度（compactness）、凹面（concavity）、凹点（concave points）、对称性（symmetry）和分形维数（fractal dimension）等。同时，为上述10个特征值分别提供了三种统计量，分别为均值（mean）、标准差（standard error）和最大值（worst or largest）

2.目的及分析任务
（1）用训练集对SVM模型进行训练；
（2）使用SVM模型对数据集中的诊断结果进行预测；
（3）对SVM模型进行评价；

3.方法及工具
实现SVM算法的Python第三方工具包有LibSVM、skearn、CVXOPT。
其中LibSVM、skearn是将SVM算法封装为接口供人们使用，具有方便快捷的优点。
本章将采用的是skearn。

'''