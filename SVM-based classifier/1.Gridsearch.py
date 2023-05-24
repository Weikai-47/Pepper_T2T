import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("./pepper.csv",sep=',')
data1 = pd.read_excel("./STab.356_reseq(1).xlsx")[["Sample name","Capsaicinoids content (mg/kg DW)"]]

index = data.columns
data = pd.DataFrame(np.array(data).T)
data.index = index
data = data.reset_index()

data.columns = data.iloc[0,:]
data = data.iloc[1:,:]

data = data.merge(data1,left_on="gene_id",right_on="Sample name")

name_list = []
for i in data.columns:
    if i != "Sample name":
        name_list.append(i)

data = data[name_list]
data['Capsaicinoids content (mg/kg DW)'] = data['Capsaicinoids content (mg/kg DW)'].astype('float')
data = data[data["Capsaicinoids content (mg/kg DW)"] >= 0]

#print(data)

#(2)数据可视化
plt.hist(data['Capsaicinoids content (mg/kg DW)'])
plt.show()

C_level = np.array(data['Capsaicinoids content (mg/kg DW)'].astype(float)).tolist()
#print(C_level)
C_level.sort(reverse=True)
#print(C_level)

print("中位数：{}".format(C_level[int(len(C_level)/2)]))
print("1/3分位点：{}".format(C_level[int(len(C_level)/3)]))
print("2/3分位点：{}".format(C_level[int(len(C_level)*2/3)]))

#（3）label划分
def func2(x):
    if x <= 220.76:
        return 0.0
    else:
        return 1.0

data['Capsaicinoids content (mg/kg DW)'] = data['Capsaicinoids content (mg/kg DW)'].apply(func2).astype(int)
data = data.set_index("gene_id")

#(4)数据集划分
X = []
Y = []

for i in range(len(data)):
    X.append(np.array(data.iloc[i, 0:119]).tolist())
    Y.append((data.iloc[i, 119]))

X = np.array(X)
Y = np.array(Y)

# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
ss = StandardScaler()
X = ss.fit_transform(X)

#(5)建模与参数选择
param_grid={
    "kernel":['linear','rbf','sigmoid'],
    "C":np.array([0.00001,0.0001,0.001,0.01,0.1,1])
}

model = SVC()

grid=GridSearchCV(model,param_grid=param_grid,cv=80)
grid.fit(X,Y)

print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_index_)

pd.DataFrame(grid.cv_results_).to_csv("GridSearch_SVM.csv")