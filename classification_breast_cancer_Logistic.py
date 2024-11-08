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
#%%                       1.业务理解
'''
利用逻辑回归算法，对病例的诊断结果进行回归分析分类预测。
'''
#%%                       2.数据读取
import pandas as pd
bc_data = pd.read_csv('D:/desktop/ML/分类分析/bc_data.csv',header=0) #'header=0'表示第0行为列名

#%%                       3.数据准备
#数据集中对乳腺癌诊断有用的数据为细胞核的10个特征值，为了将该数据值提取出来，需要删除列名为‘id’和‘diagnosis’的数据
data = bc_data.drop(['id','diagnosis'],axis = 1)
import numpy as np
y = np.array(bc_data.diagnosis)

#%%% 将分类变量diagnosis编码成数值变量，二分类用LabelEncoder()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(bc_data['diagnosis'])
#原先诊断结果diagnosis从M（恶性）和B（良性）转换成  1（恶性）和 0（良性）

#%%%划分训练集和测试集，train_test_split()，7：3
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, y_encoded, test_size=0.3, random_state=42)  

#%%%对训练集和测试集数据进行标准化StandardScaler()
# SVM 对特征的尺度敏感，因此需要对特征进行标准化。
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
x_train_scaled = scaler.fit_transform(x_train) #训练集的‘诊断学习’，得到标准差和均值
#用相同的均值和标准差来转换训练集和测试集，P.S 训练集和测试集不能拟合两次，因为这样会导致data leakage，影响模型在新数据上的泛化能力
x_test_scaled = scaler.transform(x_test)

x_train_scaled = pd.DataFrame(x_train_scaled, columns=data.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=data.columns)

#%%                       4.模型训练
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(x_train_scaled, y_train)

#%%                       5.模型评价
y_pred = lg.predict(x_test_scaled)
# 准确率、精确率、召回率和f1值
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("准确率：", accuracy_score(y_test, y_pred))
print("精确率：", precision_score(y_test, y_pred))
print("召回率：", recall_score(y_test, y_pred))
print("f1值：", f1_score(y_test, y_pred))
# 混淆矩阵:
#  [[106   2]
#  [  1  62]]
# 准确率： 0.9824561403508771
# 精确率： 0.96875
# 召回率： 0.9841269841269841
# f1值： 0.9763779527559054

#%%                       6.调参
'''
逻辑回归最常见超参数：
C（正则化参数）：控制正则化的强度。较小的值会使模型更简单，避免过拟合；较大的值会减少正则化的强度，可能导致过拟合。
solver（求解器）：用于优化目标函数的算法。常见的选择有：
    'liblinear'：适用于小数据集，使用的是坐标下降法。
    'lbfgs'、'newton-cg' 和 'sag'：适用于大数据集，使用的是拟牛顿法。
max_iter：最大迭代次数，设置较大的值可以避免在一些复杂问题上模型无法收敛。
penalty（正则化方式）：决定使用哪种正则化方式，常见的有：
    'l2'：L2 正则化（默认），适用于大多数问题。
    'l1'：L1 正则化，常用于特征选择。
multi_class：处理多类分类问题时使用的策略，常用选项有：
    'ovr'（One-vs-Rest，默认）：将多类问题分解为多个二分类问题。
    'multinomial'：适用于多项式模型。
'''
#%%%
model = LogisticRegression(max_iter=10000)
from sklearn.model_selection import GridSearchCV
# 为liblinear solver设置参数网格
param_grid_liblinear = {
    'C': [0.001, 0.01, 0.1, 1, 10, 20, 50, 100],
    'penalty': ['l1', 'l2'],  # liblinear 支持 l1 和 l2
    'multi_class': ['ovr'],    # liblinear 不支持 multinomial
    'solver': ['liblinear']
}
# liblinear solver调参
grid_search_liblinear = GridSearchCV(model, param_grid=param_grid_liblinear, cv=5, n_jobs=-1)
grid_search_liblinear.fit(x_train_scaled, y_train)
print("liblinear最佳参数:", grid_search_liblinear.best_params_)
print("最佳交叉验证准确率: {:.4f}".format(grid_search_liblinear.best_score_))
# Fitting 5 folds for each of 12 candidates, totalling 60 fits
# liblinear最佳参数: {'C': 0.1, 'multi_class': 'ovr', 'penalty': 'l2', 'solver': 'liblinear'}
# 最佳交叉验证准确率: 0.9774

# 为 lbfgs solver 设置参数网格
param_grid_lbfgs = {
    'C': [0.001, 0.01, 0.1, 1, 10, 20, 50, 100],
    'penalty': ['l2'],  # lbfgs 只支持 l2
    'multi_class': ['ovr', 'multinomial'],  # lbfgs 支持 multinomial
    'solver': ['lbfgs']
}
# lbfgs solver 调参
grid_search_lbfgs = GridSearchCV(model, param_grid=param_grid_lbfgs, cv=5, n_jobs=-1)
grid_search_lbfgs.fit(x_train_scaled, y_train)
print("lbfgs最佳参数:", grid_search_lbfgs.best_params_)
print("最佳交叉验证准确率: {:.4f}".format(grid_search_lbfgs.best_score_))
# Fitting 5 folds for each of 12 candidates, totalling 60 fits
# lbfgs最佳参数: {'C': 0.1, 'multi_class': 'multinomial', 'penalty': 'l2', 'solver': 'lbfgs'}
# 最佳交叉验证准确率: 0.9748

#%%                        7.预测，用liblinear
lg = LogisticRegression(C=0.1, penalty='l2')
lg.fit(x_train_scaled, y_train)
y_pred = lg.predict(x_test_scaled)
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("准确率：", accuracy_score(y_test, y_pred))
print("精确率：", precision_score(y_test, y_pred))
print("召回率：", recall_score(y_test, y_pred))
print("f1值：", f1_score(y_test, y_pred))
# 混淆矩阵:
#  [[108   0]
#  [  2  61]]
# 准确率： 0.9883040935672515
# 精确率： 1.0
# 召回率： 0.9682539682539683
# f1值： 0.9838709677419354

#提高了一点，但是precision=1，is it normal? 要考虑数据泄露吗？数据不平衡？过拟合？













