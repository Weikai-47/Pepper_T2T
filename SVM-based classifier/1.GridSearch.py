import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

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

# set the models
models = [
    ("Support Vector Machine (Kernal: linear)", SVC(), {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10], "kernel": [ "linear" ]}),
    ("Support Vector Machine (Kernal: poly)", SVC(), {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10], "kernel": ["poly"], "degree": [2, 3, 4, 5, 6, 7, 8, 9, 10]}),
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

scores = {}
# (3)train and evaluate the models
for name, model, params in models:
    # Grid Search the hyperparameters
    grid_search = GridSearchCV(model, params, cv=10, scoring="accuracy")
    grid_search.fit(X, y)

    # print the best hyperparameters
    print(f"{name}:")
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # use the best hyperparameters to test
    model = grid_search.best_estimator_
    scores1 = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    scores[name] = scores1

    # output the outcome
    print("Cross-Validation Scores:")
    print("Accuracy:", scores1.mean())
    print("-------------------------------")

# (4)Viualization
# Generate some random data to represent in the boxplot
np.random.seed(10)
data = pd.DataFrame(scores)

# Set the style of the seaborn library
sns.set_style("whitegrid")

# Create a figure and a set of subplots
plt.figure(figsize=(10,6))

# Create the boxplot with additional parameters for better aesthetics
sns.boxplot(data=data,
             showmeans=True,
             meanprops={"marker":"o",
                        "markerfacecolor":"white",
                        "markeredgecolor":"black",
                        "markersize":"10"})

# Set the labels and title
plt.xlabel('Prediction Algorithms', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)

# Improve the aesthetics of the plot axes
plt.tick_params(axis='both', which='major', labelsize=12)

# Show the plot
plt.show()
