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
（3）细胞核的10个特征值，包括radius（半径）、texture（纹理）、perimeter（周长）、面积（area）、平滑度（smoothness）、
    紧凑度（compactness）、凹面（concavity）、凹点（concave points）、对称性（symmetry）和分形维数（fractal dimension）等。
    同时，为上述10个特征值分别提供了三种统计量，分别为均值（mean）、标准差（standard error）和最大值（worst or largest）

2.目的及分析任务
理解机器学习方法在数据分析中的应用——KNN方法进行分类分析：
（1）以随机选择的部分记录为训练集进行学习概念——“诊断结果（diagnosis）”；
（2）以剩余记录为测试集，进行KNN建模；
（3）按KNN模型预测测试集的dignosis类型；
（4）将KNN模型给出的diagnosis“预测类型”与数据集bc_data.csv自带的“实际类型”进行对比分析，验证KNN建模的有效性。

3.方法及工具
实现KNN算法的Python第三方工具包有scikit-learn，numpy，pandas，matplotlib，seaborn，operator和os等，比较常用的是scikit-learn，numpy，pandas，matplotlib包，本章将采用的是pandas，numpy，scikit-learn，os和matplotlib包。

KNN模型的类别有暴力法、KD树和球法。
暴力法适合小数据集，但效率低下。
KD 树适合低维数据，加速效果明显，但在高维数据中会失效。
球树适合较高维数据，尤其是在数据分布不均匀时表现更好。
'''

#%%                       1.业务理解
'''
将569条数据分为训练集和测试集，通过KNN算法模型预测测试集的诊断结果，并与实际诊断结果对比分析，从而验证模型的有效性。
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

#%%%划分训练集和测试集，train_test_split()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, y_encoded, test_size=0.25, random_state=1)

#%%%对训练集和测试集数据进行标准化StandardScaler()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
x_train_scaled = scaler.fit_transform(x_train) #训练集的‘诊断学习’，得到标准差和均值
#用相同的均值和标准差来转换训练集和测试集，P.S 训练集和测试集不能拟合两次，因为这样会导致data leakage，影响模型在新数据上的泛化能力
x_test_scaled = scaler.transform(x_test)

x_train_scaled = pd.DataFrame(x_train_scaled, columns=data.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=data.columns)

#%%                       4.模型训练
#用训练集的数据学习概念‘诊断结果’，然后用测试集进行KNN建模。
from sklearn.neighbors import KNeighborsClassifier
# 使用KD树训练模型
knn_kd = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
knn_kd.fit(x_train_scaled, y_train)

#%%                       5.模型评价
# 预测并计算准确率
y_pred = knn_kd.predict(x_test_scaled)
from sklearn.metrics import accuracy_score
print("准确率：", accuracy_score(y_test, y_pred))
#准确率： 0.951048951048951，可以考虑进一步优化

#%%                       6.模型调参
#%%% k值的大小对于模型预测结果会产生很大的影响，利用准确率score()来计算K值范围在1~22的准确率
num_neigh = range(1,13)
KNNs = [KNeighborsClassifier(n_neighbors=i) for i in num_neigh]
range(len(KNNs))
scores = [KNNs[i].fit(x_train_scaled, y_train).score(x_test_scaled,y_test) for i in range(0,12)]
import matplotlib.pyplot as plt
plt.plot(num_neigh, scores)

#%%% 使用Grid Search调优KNN参数
from sklearn.model_selection import GridSearchCV
# 定义模型
knn = KNeighborsClassifier()

# 定义参数网格
param_grid = {
    'n_neighbors': list(range(1, 16)),  # 尝试1到15个邻居
    'weights': ['uniform', 'distance'],  # 权重方式
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # 选择算法
    'p': [1, 2]  # 曼哈顿距离和欧氏距离
}

# 使用 GridSearchCV 进行搜索
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')  # 使用5折交叉验证
grid_search.fit(x_train_scaled, y_train)

# 输出最佳参数组合
print("最佳参数组合:", grid_search.best_params_)
print("最佳准确率:", grid_search.best_score_)
#最佳参数：{'algorithm': 'auto', 'n_neighbors': 3, 'p': 1, 'weights': 'uniform'}
#最佳准确率： 0.9764979480164158

#%%% 重新拟合
knn_3= KNeighborsClassifier(n_neighbors=3)
knn_3.fit(x_train_scaled, y_train)

# 预测并计算准确率
y_pred = knn_3.predict(x_test_scaled)
print("准确率：", accuracy_score(y_test, y_pred))
#准确率： 0.958041958041958，调参后比原模型的准确度提高一点。

#%%                       7.模型预测
# 任意输入一个样本并预测样本的诊断结果，以第421个数据样本为例
x_test_scaled.loc[[5]]
y_predict_421 = knn_3.predict(x_test_scaled.)









[       nan 0.96240766        nan 0.95543092        nan 0.96240766
        nan 0.95543092        nan 0.97649795        nan 0.9671409
        nan 0.97649795        nan 0.96949384        nan 0.96943912
        nan 0.96010944        nan 0.97414501        nan 0.96478796
        nan 0.96708618        nan 0.96949384        nan 0.97179207
        nan 0.9671409         nan 0.96010944        nan 0.96481532
        nan 0.96711354        nan 0.96246238        nan 0.96478796
        nan 0.96248974        nan 0.96711354        nan 0.9601368
        nan 0.96010944        nan 0.96481532        nan 0.95307798
        nan 0.96246238        nan 0.95307798        nan 0.95781122
 0.96240766 0.96240766 0.95543092 0.95543092 0.95540356 0.96240766
 0.95543092 0.95543092 0.97649795 0.97649795 0.9671409  0.9671409
 0.97414501 0.97649795 0.9671409  0.96949384 0.96943912 0.96943912
 0.96010944 0.96010944 0.97179207 0.97414501 0.96478796 0.96478796
 0.96708618 0.96708618 0.96949384 0.96949384 0.96478796 0.97179207
 0.95778386 0.9671409  0.96010944 0.96010944 0.96248974 0.96481532
 0.95772914 0.96711354 0.9601368  0.96246238 0.96478796 0.96478796
 0.95781122 0.96248974 0.95540356 0.96711354 0.95543092 0.9601368
 0.95307798 0.96010944 0.95778386 0.96481532 0.94837209 0.95307798
 0.94837209 0.96246238 0.94837209 0.95307798 0.95545828 0.95781122
 0.96240766 0.96240766 0.95543092 0.95543092 0.95540356 0.96240766
 0.95543092 0.95543092 0.97649795 0.97649795 0.9671409  0.9671409
 0.97414501 0.97649795 0.9671409  0.96949384 0.96943912 0.96943912
 0.96010944 0.96010944 0.97179207 0.97414501 0.96478796 0.96478796
 0.96708618 0.96708618 0.96949384 0.96949384 0.96478796 0.97179207
 0.95778386 0.9671409  0.96010944 0.96010944 0.96248974 0.96481532
 0.95772914 0.96711354 0.9601368  0.96246238 0.96478796 0.96478796
 0.95781122 0.96248974 0.95540356 0.96711354 0.95543092 0.9601368
 0.95307798 0.96010944 0.95778386 0.96481532 0.94837209 0.95307798
 0.94837209 0.96246238 0.94837209 0.95307798 0.95545828 0.95781122
        nan 0.96240766        nan 0.95543092        nan 0.96240766
        nan 0.95543092        nan 0.97649795        nan 0.9671409
        nan 0.97649795        nan 0.96949384        nan 0.96943912
        nan 0.96010944        nan 0.97414501        nan 0.96478796
        nan 0.96708618        nan 0.96949384        nan 0.97179207
        nan 0.9671409         nan 0.96010944        nan 0.96481532
        nan 0.96711354        nan 0.96246238        nan 0.96478796
        nan 0.96248974        nan 0.96711354        nan 0.9601368
        nan 0.96010944        nan 0.96481532        nan 0.95307798
        nan 0.96246238        nan 0.95307798        nan 0.95781122]


































