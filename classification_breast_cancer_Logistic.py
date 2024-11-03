'''
分类分析（classification）————病例自动判断分析

1.数据及分析对象
'bc_data.csv'，数据内容来自“威斯康星乳腺癌数据库（Wisconsin Breast Cancer Database）” ，该数据集主要记录了569个病例的32个属性。主要属性/字段如下：
（1）ID：病例的ID；
（2）Diagnosis（诊断结果）：M为恶性，B为良性。该数据集共包含357个良性病例和212 个恶性病例；
（3）细胞核的10个特征值，包括radius（半径）、texture（纹理）、perimeter（周长）、面积（area）、平滑度（smoothness）、紧凑度（compactness）、凹面（concavity）、凹点（concave points）、对称性（symmetry）和分形维数（fractal dimension）等。同时，为上述10个特征值分别提供了三种统计量，分别为均值（mean）、标准差（standard error）和最大值（worst or largest）

2.目的及分析任务
理解机器学习方法在数据分析中的应用——采用逻辑回归方法进行分类分析：
（1）划分训练集与测试集，利用逻辑回归算法进行模型训练，分类分析；
（2）进行模型评价，调整模型参数；
（3）按调参后的模型进行模型预测，得出的结果与测试集结果进行对比分析，验证逻辑回归的有效性；

3.方法及工具
采用的是sklearn.linear_model.LogisticRegression。

'''