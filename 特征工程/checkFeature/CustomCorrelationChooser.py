from copy import deepcopy

from sklearn.base import TransformerMixin,BaseEstimator
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

'''
    通过corr函数筛选出两列之间相关性系数的绝对值大于某个阈值的字段名称
'''
def messagePrint(x):
    print(x)
    print('----------------------------------------')

class CustomCorrelationChooser(TransformerMixin,BaseEstimator):
    def __init__(self,response,cols_to_keep=[],threshold=None):
        # 保存响应变量
        self.response = response
        # 保存阈值
        self.threshold = threshold
        # 初始化一个变量，存放要保留的列名
        self.cols_to_keep = cols_to_keep

    def transform(self,X):
        # 转换-选择合适的列
        return X[self.cols_to_keep]

    def fit(self,X,*_):
        # 创建新的DataFrame,存放特征和响应
        df = pd.concat([X,self.response],axis=1)
        # 保存高于阈值的列的名称
        self.cols_to_keep = df.columns[df.corr()[df.columns[-1]].abs() > self.threshold]
        # 只保留X的列，删除响应变量
        self.cols_to_keep = [c for c in self.cols_to_keep if c in X.columns]
        return self

def get_best_model_and_accuracy(name,model,params,X,y):
    grid = GridSearchCV(model, # 要搜索的模型
                        params, # 要尝试的参数
                        error_score=0, # 如果报错，则结果为零
                        cv=5
                        )
    # 管道设计
    '''
    mean_impute = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                            ('classify', knn)
                            ])
    grid = GridSearchCV(mean_impute, knn_params)
    '''

    grid.fit(X,y) # 拟合模型和参数
    # 经典的性能指示
    print(name + ' - Best Accuracy : {}'.format(grid.best_score_))
    # 得到最佳准确率的最佳参数
    print(name + ' - Best Parameters : {}'.format(grid.best_params_))
    # 拟合的平均时间（秒）
    print(name + " - Average Time to Fit (s) ：{}".format(round(grid.cv_results_['mean_fit_time'].mean(),3)))
    # 预测的平均时间 （秒）,从该指标可以看出模型在真实世界的性能
    print(name + " - Average Time to Score (s) : {}".format(round(grid.cv_results_['mean_score_time'].mean(),3)))

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
'''
基于决策树的特征选择器
'''
def chooseFeatureByDecisionTree(model,X,y):
    from sklearn.feature_selection import SelectFromModel
    # 实例化一个类，按照决策树分类器的内部指标排序重要性，选择特征
    #select_from_model = SelectFromModel(DecisionTreeClassifier(),threshold=0.5)
    # 拟合数据
    #selected_X = select_from_model.fit_transform(X,y)
    #selected_X.shape

    # 为后面加速
    tree_pipe_params = {'max_depth':[1,3,5,7]}
    #创建基于DecisionTreeClassifier的SelectFromModel
    select = SelectFromModel(DecisionTreeClassifier())
    select_from_pipe = Pipeline([('select',select),
                                 ('classifier',model)])
    select_from_pipe_params = deepcopy(tree_pipe_params)
    ## 也可以增加决策树的其他参数列表
    select_from_pipe_params.update({'max_depth':[None,1,3,5,7]
                                    })
    messagePrint(select_from_pipe_params)
    get_best_model_and_accuracy('DecisionTreeClassifier',model,select_from_pipe_params,X,y)
    # 拟合数据
    select_from_pipe.steps[0][1].fit(X,y)
    # 列出选择的列
    messagePrint(X.columns[select_from_pipe.steps[0][1].get_support()])

if __name__ == '__main__':
    x,y = method01()
    print(x.info())
    ccc = CustomCorrelationChooser(threshold=0.2,response=y)
    ccc.fit(x)
    print(ccc.cols_to_keep)
    print(ccc.transform(x).head())
