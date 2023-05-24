from sklearn.feature_selection import RFECV
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
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics

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
# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
list_num = []
list_line = []
length = 1

for i in range(1,200):
    gene_list = pd.read_csv('SVM_features_select.csv')
    gene_list.columns = ['num','rank']
    gene_list = gene_list[gene_list['rank'] <= i]

    X = []
    Y = []

    for j in range(len(data)):
        X.append(np.array(data.iloc[j, gene_list['num']]).tolist())
        Y.append((data.iloc[j, 119]))

    X = np.array(X)
    Y = np.array(Y)

    ss = StandardScaler()
    X = ss.fit_transform(X)

    list_num.append(len(gene_list))

    #(5)建模与特征选择
    loo = LeaveOneOut()
    loo.get_n_splits(X)

    SVC_list = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
        # ss = StandardScaler()
        # X_train = ss.fit_transform(X_train)
        # X_test = ss.transform(X_test)

        model = SVC(C=0.01,kernel='linear')
        # 用训练集训练：
        model.fit(X_train, y_train)
        # 用测试集预测：
        prediction = model.predict(X_test)
        # print('准确率：', metrics.accuracy_score(prediction, y_test))
        SVC_list.append(metrics.accuracy_score(prediction, y_test))

    print(i)
    print(sum(SVC_list) / len(SVC_list))

    list_line.append(sum(SVC_list) / len(SVC_list))

list_num = pd.DataFrame(list_num).to_csv('SVM_list_num_linear.csv',index=False)
list_line = pd.DataFrame(list_line).to_csv('SVM_list_line_linear.csv',index=False)

