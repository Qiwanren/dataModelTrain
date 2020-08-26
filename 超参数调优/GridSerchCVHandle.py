from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

def method01():
    np.random.seed(123)
    path = 'D:/data/python/test/credit_card_default.csv'

    # 导入数据集
    credit_card_default = pd.read_csv(path)
    # 进行基础数据的信息分析
    print(credit_card_default.shape)
    #messagePrint(credit_card_default.describe().T)
    # 特征矩阵
    X = credit_card_default.drop('default payment next month',axis=1)
    # 响应变量
    y = credit_card_default['default payment next month']
    # 输出空准确率
    #print(y.value_counts(normalize=True))
    return X,y

def get_best_model_and_accuracy(name,model,params,X,y):
    grid = GridSearchCV(model, # 要搜索的模型
                        params, # 要尝试的参数
                        error_score=0 # 如果报错，则结果为零
                        )
    grid.fit(X,y) # 拟合模型和参数
    # 经典的性能指示
    print(name + ' - Best Accuracy : {}'.format(grid.best_score_))
    # 得到最佳准确率的最佳参数
    print(name + ' - Best Parameters : {}'.format(grid.best_params_))
    # 拟合的平均时间（秒）
    print(name + " - Average Time to Fit (s) ：{}".format(round(grid.cv_results_['mean_fit_time'].mean(),3)))
    # 预测的平均时间 （秒）,从该指标可以看出模型在真实世界的性能
    print(name + " - Average Time to Score (s) : {}".format(round(grid.cv_results_['mean_score_time'].mean(),3)))

if __name__ == '__main__':
    x,y = method01()
    # 随机森林
    forest_params = {'n_estimators': [10, 50, 100], 'max_depth': [None, 1, 3, 5, 7]}
    forest = RandomForestClassifier()
    x, y = method01()
    # 预测模型
    get_best_model_and_accuracy('RandomForestClassifier', forest, forest_params, x, y)


