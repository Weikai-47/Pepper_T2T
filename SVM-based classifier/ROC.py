import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# (1) load and transform the data
data = pd.read_csv("../capsaicin_cnv_for_mengtan.csv", sep=',')
data1 = pd.read_excel("../STab.356_reseq(1).xlsx")[["Sample name","Capsaicinoids content (mg/kg DW)"]]

index = data.columns
data = pd.DataFrame(np.array(data).T)
data.index = index
data = data.reset_index()

data.columns = data.iloc[1,:]
data = data.iloc[2:,:]
data = data.merge(data1,left_on="geneName",right_on="Sample name")

name_list = []
for i in data.columns:
    if i != "Sample name":
        name_list.append(i)

data = data[name_list]
list1 = [i for i in data.columns]
list1[0] = 'SampleID'
data.columns = list1
data['Capsaicinoids content (mg/kg DW)'] = data['Capsaicinoids content (mg/kg DW)'].astype('float')
data = data[data["Capsaicinoids content (mg/kg DW)"] >= 0]

# split the labels according to the median, so we can get the balanced dataset
C_level = np.array(data['Capsaicinoids content (mg/kg DW)'].astype(float)).tolist()
C_level.sort(reverse=True)

def func2(x):
    if x <= C_level[int(len(C_level)/2)]:
        return 0.0
    else:
        return 1.0

data['Capsaicinoids content (mg/kg DW)'] = data['Capsaicinoids content (mg/kg DW)'].apply(func2).astype(int)

# (2) split the train dataset and test dataset
X = data.drop(["SampleID", "Capsaicinoids content (mg/kg DW)"], axis=1)
y = data["Capsaicinoids content (mg/kg DW)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (3) 建立model，绘制ROC曲线
SVM_model = SVC(C=0.1,kernel='linear')
RF_model = RandomForestClassifier(max_depth=1,max_features='sqrt',n_estimators=150)
KNN_model = KNeighborsClassifier(n_neighbors=7,weights='uniform')
GBoost_model = GradientBoostingClassifier(learning_rate=0.1,max_depth=30,max_features='sqrt',n_estimators=50,subsample=0.8)

# 训练模型
SVM_model.fit(X_train, y_train)
RF_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)
GBoost_model.fit(X_train, y_train)

# 计算各个模型的概率预测值并绘制ROC曲线
models = [SVM_model, RF_model, KNN_model, GBoost_model]
model_names = ['SVM', 'Random Forest', 'K-Nearest Neighbors', 'Gradient Boosting']

plt.figure()
for model, model_name in zip(models, model_names):
    if isinstance(model, RandomForestClassifier) or isinstance(model, KNeighborsClassifier):
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # 使用predict_proba获取概率值
    else:
        y_pred_proba = model.decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.show()