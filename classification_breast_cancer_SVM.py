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
#%%                       1.业务理解
'''
训练SVM模型，对病例的诊断结果进行分类预测。
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
x_train, x_test, y_train, y_test = train_test_split(data, y_encoded, test_size=0.2, random_state=1)  

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
from sklearn.svm import SVC
# 用线性SVM
svm = SVC(kernel='linear', C=0.2)
svm.fit(x_train_scaled, y_train)

#%%                       5.模型评价
y_pred = svm.predict(x_test_scaled)
# 准确率、精确率、召回率和f1值
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("准确率：", accuracy_score(y_test, y_pred))
print("精确率：", precision_score(y_test, y_pred))
print("召回率：", recall_score(y_test, y_pred))
print("f1值：", f1_score(y_test, y_pred))
# 混淆矩阵:
#  [[71  1]  FN = 1：有 1 个恶性样本被误判为良性。
#  [ 3 39]]  FP = 3：有 3 个良性样本被误判为恶性。
# 准确率： 0.9649122807017544
# 精确率： 0.975
# 召回率： 0.9285714285714286
# f1值： 0.951219512195122

#%%% ROC 曲线和 AUC 值
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_prob = svm.decision_function(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='SVM (AUC = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#%%                       6.调参
model = SVC(kernel='linear')

# 5. 使用 Grid Search 调参
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 0.2, 0.3, 0.4, 0.5]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train_scaled, y_train)
print("最佳参数: ", grid_search.best_params_)
print("最佳交叉验证准确率: {:.4f}".format(grid_search.best_score_))

# 最佳参数:  {'C': 0.2}
# 最佳交叉验证准确率: 0.9758









 