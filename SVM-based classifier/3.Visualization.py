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

def Visualize_Performance(acc_scores):
    # Generate some random data to represent in the boxplot
    np.random.seed(10)
    data = pd.DataFrame(acc_scores)

    # Set the style of the seaborn library
    sns.set_style("whitegrid")

    # Create a figure and a set of subplots
    plt.figure(figsize=(10, 6))

    # Create the boxplot with additional parameters for better aesthetics
    sns.boxplot(data=data,
                showmeans=True,
                meanprops={"marker": "o",
                           "markerfacecolor": "white",
                           "markeredgecolor": "black",
                           "markersize": "10"})

    # Set the labels and title
    plt.xlabel('Prediction Algorithms', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    # Improve the aesthetics of the plot axes
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Show the plot
    plt.show()

def Draw_ROC_curve(X,Y,model_list,random_state):
    # (1) split the train dataset and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    # (2) 建立model，绘制ROC曲线
    SVM_model = model_list[0]
    RF_model = model_list[1]
    KNN_model = model_list[2]
    GBoost_model = model_list[3]

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
    
