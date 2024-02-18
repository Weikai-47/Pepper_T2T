import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def stat_feature_importance(X, Y, model, name_list):
    X = np.array(X)
    Y = np.array(Y)

    # 训练模型
    model.fit(X, Y)

    # 根据模型类型获取特征重要性
    if hasattr(model, 'coef_'):
        # 对于线性模型
        feature_importance = np.abs(model.coef_)
    elif hasattr(model, 'feature_importances_'):
        # 对于支持feature_importances_属性的模型，如RF和Gradient Boosting
        feature_importance = model.feature_importances_
    else:
        raise ValueError("Model does not have feature importance attribute")

    # 对于线性模型，feature_importance是一个二维数组，我们需要将其转化为一维
    if len(feature_importance.shape) > 1:
        feature_importance = np.mean(feature_importance, axis=0)

    # 创建一个字典来存储特征名称和它们的重要性
    feature_importance_dict = dict(zip(name_list, feature_importance))
    # 按重要性降序排序字典
    sorted_feature_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

    with open("../features_importance.csv", "w") as f:
        # 输出排序后的特征重要性
        for feature, importance in sorted_feature_importance.items():
            print(f"{feature}: {importance}")
            f.write(feature + "," + str(importance) + '\n')


