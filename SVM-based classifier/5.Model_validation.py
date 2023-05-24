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
#plt.show()

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
list111 = pd.read_csv("SVM_features_select.csv")
list111.columns = ["1","2"]
list112 = list111[list111["2"]<=48]["1"]
list113 = []
for i in list112:
    list113.append(i)

X = []
Y = []

for i in range(len(data)):
    X.append(np.array(data.iloc[i, list113]).tolist())
    Y.append((data.iloc[i, 119]))

X = np.array(X)
Y = np.array(Y)

ss = StandardScaler()
X = ss.fit_transform(X)

#(5)建模与参数选择
clf = SVC(C=0.01,kernel='linear')
clf.fit(X,Y)

data = pd.read_excel("./sangping.xls")
index = data.columns
data = pd.DataFrame(np.array(data).T)
data.index = index
data = data.reset_index()
print(data)
X = []
for i in range(len(data)):
    X.append(np.array(data.iloc[i, list113]).tolist())

X = np.array(X)
Y_predict = clf.predict(X)
print(Y_predict)
