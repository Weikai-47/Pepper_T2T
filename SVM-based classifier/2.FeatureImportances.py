from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

data.to_csv('data.csv',index=False)

# (2) split the train dataset and test dataset
X = data.drop(["SampleID", "Capsaicinoids content (mg/kg DW)"], axis=1)
y = data["Capsaicinoids content (mg/kg DW)"]

gene_list = X.columns

X = np.array(X)
Y = np.array(y)

# (3) output the importances of the model
svm_model = SVC(C=0.1,kernel='linear')
svm_model.fit(X, Y)

# get the importances of features
feature_importance = np.abs(svm_model.coef_)

# create a dictionary to store feature names and their importances
feature_importance_dict = dict(zip(gene_list, feature_importance[0]))

# sort the dictionary by importance in descending order
sorted_feature_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

# output the sorted feature importances
for feature, importance in sorted_feature_importance.items():
    print(f"{feature}: {importance}")

# Visualization with the word cloud
# 创建一个以特征名称为词的字典
wordcloud_data = {feature: importance for feature, importance in sorted_feature_importance.items()}

# 创建词云对象
wordcloud = WordCloud(width=800, height=400, background_color='white')

# 生成词云
wordcloud.generate_from_frequencies(wordcloud_data)

# 绘制词云
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()