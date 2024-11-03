'''
分类分析（classification）————病例自动判断分析
本节分别采用4种不同的方法实现病例自动化判断。
1）贝叶斯分类算法：scikit-learn的naive_bayes
2)KNN算法：scikit-learn
3)SVM算法：LibSVM,scikit-learn,CVXOPT;LibSVM和scikit的功能是将SVM算法封装为接口供使用，方便快捷。而CVXPOT则是主要用于解决凸优化问题，因为SVM可看作凸优化问题的求解
4）逻辑回归算法：scikit-learn的linear_model.LogisticRegression


1.数据及分析对象
'bc_data.csv'，数据内容来自“威斯康星乳腺癌数据库（Wisconsin Breast Cancer Database）” ，该数据集主要记录了569个病例的32个属性。主要属性/字段如下：
（1）ID：病例的ID；
（2）Diagnosis（诊断结果）：M为恶性，B为良性。该数据集共包含357个良性病例和212 个恶性病例；
（3）细胞核的10个特征值，包括radius（半径）、texture（纹理）、perimeter（周长）、面积（area）、平滑度（smoothness）、紧凑度（compactness）、凹面（concavity）、凹点（concave points）、对称性（symmetry）和分形维数（fractal dimension）等。同时，为上述10个特征值分别提供了三种统计量，分别为均值（mean）、标准差（standard error）和最大值（worst or largest）

2.目的及分析任务
理解机器学习方法在数据分析中的应用——KNN方法进行分类分析：
（1）以随机选择的部分记录为训练集进行学习概念——“诊断结果（diagnosis）”；
（2）以剩余记录为测试集，进行KNN建模；
（3）按KNN模型预测测试集的dignosis类型；
（4）将KNN模型给出的diagnosis“预测类型”与数据集bc_data.csv自带的“实际类型”进行对比分析，验证KNN建模的有效性。

3.方法及工具
（1）实现贝叶斯分类算法的Python第三方工具包有scikit-learn，本章将采用的是scikit-learn中的naive_bayes模块。
（2）实现KNN算法的Python第三方工具包有scikit-learn，numpy，pandas，matplotlib，seaborn，operator和os等，比较常用的是scikit-learn，numpy，pandas，matplotlib包，本章将采用的是pandas，numpy，scikit-learn，os和matplotlib包。
（3）实现SVM算法的Python第三方工具包有LibSVM、skearn、CVXOPT。其中LibSVM、skearn是将SVM算法封装为接口供人们使用，具有方便快捷的优点。本章将采用的是skearn。
（4）实现逻辑回归算法的Python第三方工具包有scikit-learn等，本章将采用的是sklearn.linear_model.LogisticRegression。

'''