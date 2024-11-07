'''
分类分析（classification）————病例自动判断分析

1.数据及分析对象
'bc_data.csv'，数据内容来自“威斯康星乳腺癌数据库（Wisconsin Breast Cancer Database）” ，该数据集主要记录了569个病例的32个属性。主要属性/字段如下：
（1）ID：病例的ID；
（2）Diagnosis（诊断结果）：M 为恶性，B 为良性。该数据集共包含357个良性病例和212 个恶性病例；
（3）细胞核的10个特征值，包括radius（半径）、texture（纹理）、perimeter（周长）、面积（area）、平滑度（smoothness）、紧凑度（compactness）、凹面（concavity）、凹点（concave points）、对称性（symmetry）和分形维数（fractal dimension）等。同时，为上述10个特征值分别提供了三种统计量，分别为均值（mean）、标准差（standard error）和最大值（worst or largest）

2.目的及分析任务
理解机器学习方法在数据分析中的应用——用朴素贝叶斯算法进行分类分析：
（1）以一定比例将数据集划分为训练集和测试集；
（2）利用训练集进行朴素贝叶斯算法的建模；
（3）使用朴素贝叶斯分类模型在测试集上对诊断结果进行预测；
（4）将朴素贝叶斯分类模型对诊断结果的分类预测与真实的诊断结果进行对比分析，验证模型的有效性。

3.方法及工具
scikit-learn中的naive_bayes模块。

'''
#%%                       1.业务理解
'''
使用朴素贝叶斯分类算法，根据病例数据集中的特征属性，对病例的诊断结果进行分类预测。
'''
#%%                       2.数据读取
import pandas as pd
bc_data = pd.read_csv('D:/desktop/ML/分类分析/bc_data.csv',header=0) #'header=0'表示第0行为列名

#%%                       3.数据准备
#数据集中对乳腺癌诊断有用的数据为细胞核的10个特征值，为了将该数据值提取出来，需要删除列名为‘id’和‘diagnosis’的数据
data = bc_data.drop(['id','diagnosis'],axis = 1)
import numpy as np
y = np.array(bc_data.diagnosis)

#%%% 将分类变量diagnosis编码成数值变量
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(bc_data['diagnosis'])
#原先诊断结果diagnosis从M（恶性）和B（良性）转换成  1（恶性）和 0（良性）

#%%%划分训练集和测试集，train_test_split()，7：3
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, y_encoded, test_size=0.3, random_state=40,stratify=y)  #'stratify=y'表示数据集划分后的样本分布与划分前的分布相同。

#%%%对训练集和测试集数据进行标准化StandardScaler()
# TIP 先跳过，书中没用scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
x_train_scaled = scaler.fit_transform(x_train) #训练集的‘诊断学习’，得到标准差和均值
#用相同的均值和标准差来转换训练集和测试集，P.S 训练集和测试集不能拟合两次，因为这样会导致data leakage，影响模型在新数据上的泛化能力
x_test_scaled = scaler.transform(x_test)

x_train_scaled = pd.DataFrame(x_train_scaled, columns=data.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=data.columns)

# #scaled后的代码和结果：
# from sklearn.naive_bayes import GaussianNB 
# gnb = GaussianNB()
# gnb.fit(x_train_scaled, y_train)
# y_pred = gnb.predict(x_test_scaled)
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# print("准确率：", accuracy_score(y_test, y_pred))
# print("精确率：", precision_score(y_test, y_pred))
# print("召回率：", recall_score(y_test, y_pred))
# print("f1值：", f1_score(y_test, y_pred))
# 准确率： 0.9415204678362573
# 精确率： 0.9655172413793104
# 召回率： 0.875
# f1值： 0.9180327868852458
#%%                       4.模型训练
#scikit-learn包中naive_bayes模块里根据特征类型和分布提供了多个不同的模型，其中：
# GaussianNB 假设数据符合正态分布，适用于连续值较多的特征。
# BernoulliNB 适用于二元离散值的特征。
# MultinomialNB 适用于多元离散值的特征。
# bc_data的数据特征为连续变量，所以用GaussianNB建模

from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB()
gnb.fit(x_train, y_train)

#%%                       5.模型评价
y_pred = gnb.predict(x_test)
# 准确率、精确率、召回率和f1值
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("准确率：", accuracy_score(y_test, y_pred))
print("精确率：", precision_score(y_test, y_pred))
print("召回率：", recall_score(y_test, y_pred))
print("f1值：", f1_score(y_test, y_pred))
# 准确率： 0.935672514619883
# 精确率： 0.9649122807017544
# 召回率： 0.859375
# f1值： 0.9090909090909092          

#%%                       6.调参
'''
朴素贝叶斯模型主要基于概率分布假设，不需要大量的超参数调整。
GaussionNB有两个参数priors和var_smoothing；

priors 用于设置每个类别的先验概率，当我们对某些类别有先验知识或有数据不平衡时可以调整 priors。如果不设置，默认会根据数据中的类别比例自动计算先验概率。

var_smoothing用于添加一个较小的常数值到每个特征的方差中，以防止零方差的问题。增大该值会提高模型的稳定性，特别是当数据较少或方差较小时。值越小，模型越贴合数据（但容易过拟合）。
'''
#%%% 使用 GridSearchCV 调参
from sklearn.model_selection import GridSearchCV

gnb = GaussianNB()

# 设置参数网格
param_grid = {
    'var_smoothing': np.logspace(-12, -6, num=7)  # 从 10^-12 到 10^-6 的平滑参数
}

# 使用GridSearchCV进行网格搜索
grid_search = GridSearchCV(gnb, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

# 输出最佳参数和得分
print("最佳参数：", grid_search.best_params_)
print("最佳得分：", grid_search.best_score_)
# 最佳参数： {'var_smoothing': 1e-12}
# 最佳得分： 0.9471835443037975

#%%                      7.模型预测
# 用最佳模型进行预测，grid_search.best_estimator_会返回调参后自动选择的最佳模型，直接用它来fit和predict
y_pred_tuned = grid_search.best_estimator_.predict(x_test)
# 准确率、精确率、召回率和f1值
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("准确率：", accuracy_score(y_test, y_pred_tuned))
print("精确率：", precision_score(y_test, y_pred_tuned))
print("召回率：", recall_score(y_test, y_pred_tuned))
print("f1值：", f1_score(y_test, y_pred_tuned))
# 准确率： 0.9415204678362573
# 精确率： 0.9655172413793104
# 召回率： 0.875
# f1值： 0.9180327868852458

from sklearn.metrics import accuracy_score, confusion_matrix
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
# 混淆矩阵:
#  [[105   2]
#  [  9  55]]
# TP = 105：有 105 个恶性肿瘤样本被正确预测为恶性。
# FN = 2：有 2 个恶性样本被误判为良性。
# FP = 9：有 9 个良性样本被误判为恶性。
# TN = 55：有 55 个良性样本被正确预测为良性。



















