import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import Visualization as vis
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from Feature_Importance import stat_feature_importance

# (1) load and transform the data
data = pd.read_excel("../CBGs_cnv.xlsx")
data1 = pd.read_excel("../STab.356_reseq.xlsx")[["Sample name","Capsaicinoids content (mg/kg DW)"]]

index = data.columns
data = pd.DataFrame(np.array(data).T)
data.index = index
data = data.reset_index()
data.columns = data.iloc[1,:]
data = data.iloc[2:,:]
data = data.merge(data1,left_on="Gene",right_on="Sample name")

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

# (2) Features selection
X = data.drop(["SampleID", "Capsaicinoids content (mg/kg DW)"], axis=1)
y = data["Capsaicinoids content (mg/kg DW)"]

# 初始化一个SVM分类器
RF = RandomForestClassifier()
# 使用RFECV进行特征选择，以找到最佳特征数量
# StratifiedKFold用于确保每个类的样本比例保持一致
# step表示每次迭代要移除的特征数
# cv代表交叉验证的策略，这里使用5折交叉验证
rfecv = RFECV(estimator=RF, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X, y)

# 打印出最佳特征数量
print("Optimal number of features : %d" % rfecv.n_features_)

# 选择特征
X_new = rfecv.transform(X)

# 获取被选中的特征的布尔掩码
selected_features_mask = rfecv.support_

# 使用布尔掩码来获取被选中的特征名称
selected_features_names = X.columns[selected_features_mask]

print("Selected features names:")
print(selected_features_names)

# (3) Set the parameters for models
models = [
    ("Support Vector Machine (Kernal: linear)", SVC(), {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10], "kernel": [ "linear" ]}),
    ("Random Forest", RandomForestClassifier(), {"n_estimators": [10, 50, 100, 150], "max_depth": [1, 10, 20, 30], 'max_features':[None, 'sqrt', 'log2']}),
    ("K Nearest Neighbors", KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 10], "weights": ["uniform", "distance"]}),
    ("Gradient Boosting", GradientBoostingClassifier(),
     {
         'n_estimators': [10, 50, 100, 150],
         'learning_rate': [0.01, 0.1, 0.2],
         'max_depth': [1, 10, 20, 30],
         'max_features': [None, 'sqrt', 'log2'],
         'subsample': [0.8, 0.9, 1.0]
     }
    )
]

acc_scores = {}
precision_scores = {}
recall_scores = {}

model_list = []
best_acc = 0
best_model = None

# (4) train and evaluate the models
for name, model, params in models:
    # Grid Search the hyperparameters
    grid_search = GridSearchCV(model, params, cv=10, scoring="accuracy")
    grid_search.fit(X_new, y)

    # print the best hyperparameters
    print(f"{name}:")
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # use the best hyperparameters to test
    model = grid_search.best_estimator_
    model_list.append(model)
    scores1 = cross_val_score(model, X_new, y, cv=10, scoring="accuracy")
    acc_scores[name] = scores1

    # output the outcome
    print("Cross-Validation Scores:")
    print("Accuracy:", scores1.mean())

    if scores1.mean() > best_acc:
        best_acc = scores1.mean()
        best_model = model

    # Calculate precision and recall
    scores2 = cross_val_score(model, X_new, y, cv=10, scoring="precision")
    precision_scores[name] = scores2

    # output the outcome
    print("Cross-Validation Scores:")
    print("Precision:", scores2.mean())

    scores3 = cross_val_score(model, X_new, y, cv=10, scoring="recall")
    recall_scores[name] = scores3

    # output the outcome
    print("Cross-Validation Scores:")
    print("Recall:", scores3.mean())

    print("-------------------------------")

# (5) Viualization
vis.Visualize_Performance(acc_scores)

vis.Draw_ROC_curve(X_new,y,model_list,89)

# (6) Statistic Feature importance
stat_feature_importance(X_new,y,best_model,selected_features_names)
