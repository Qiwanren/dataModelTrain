
'''
输入参数为模型名称，模型实例，分析参数值，X，y数据域

分类特征：从SelectKBest开始，用卡方或者基于树的选择器
定量特征：用线性模型和基于相关性的选择器
二元分类：使用SelectFromModel和SVC
'''
from copy import deepcopy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def messagePrint(x):
    print(x)
    print('----------------------------------------')

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

def filedHandle(x):
    return round(x,8)

'''
根据特征直接的相关性选择特征
    输入：count_rule ：评估规则，可选取值为f_classif，f_regression，chi2
            from sklearn.feature_selection import f_classif     # 用于分类特征分析
            from sklearn.feature_selection import f_regression  # 用于连续特征分析
            from sklearn.feature_selection import chi2          # 卡方检验
         X : 特征数据矩阵
         y : 标签矩阵
    结果P值的意义:假设特征与响应变量之间没有关系的概率，因此p值越低，则特征与响应变量的关系越大
         
'''
def chooseFeatureBy_P_value(count_rule,X,y):
    #selectKBest在给定目标函数后选择k个最高分
    # 只保留最佳的五个特征，
    k_best = SelectKBest(count_rule,k=15)
    #k_best = SelectKBest(f_regression,k=5)
    #k_best = SelectKBest(chi2,k=5)

    # 选择最佳特征的矩阵
    df1 = k_best.fit_transform(X,y)  # 获取选定的结果数据集

    # 获取列的p值
    #messagePrint(k_best.pvalues_)
    #messagePrint(df1)
    #特征和p值组成DataFrame
    #按p值排列
    p_values = pd.DataFrame({'column':X.columns,'p_value':k_best.pvalues_}).sort_values('p_value')
    p_values['p_value'] = p_values['p_value'].apply(lambda x: filedHandle(x))
    #选择阈值为0.05的特征
    p_values = p_values[p_values['p_value'] < 0.05]
    # 取前五个特征
    messagePrint(p_values)
    return df1


'''
基于决策树的特征选择器
    传入参数为：模型实例，数据集 X，y 
    输出最佳参数组合
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
    # 设置流水线最佳参数
    select_from_pipe.set_params(**{'classifier__max_depth': 3})
    # 拟合数据
    select_from_pipe.steps[0][1].fit(X,y)
    # 列出选择的列
    messagePrint(X.columns[select_from_pipe.steps[0][1].get_support()])

# 线性模型和正则化
def zehAndRegression(model,X,y):
    #model.fit(X,y)
    tree_pipe_params = {'classifier__max_depth': [1, 3, 5, 7]}
    # 用正则化后的逻辑回归进行选择
    logistic_selector = SelectFromModel(LogisticRegression())
    # 新流水线，用LogisticRegression的参数进行排列
    regularization_pipe = Pipeline([('select',logistic_selector),
                                    ('classifier',model)])
    regularization_pipe_params = deepcopy(tree_pipe_params)
    # L1 和 L2正则化
    regularization_pipe_params.update({
        'select__threshold':[0.01,0.05,0.1,"mean","median","2.*mean"],
        'classifier__max_depth':[1,3,5,7],
        'select__estimator__penalty':['l1', 'l2', 'elasticnet', 'none']
    })

    messagePrint(regularization_pipe_params)
    get_best_model_and_accuracy('LogisticRegression',regularization_pipe,regularization_pipe_params,X,y)

'''
数据降维操作
    1、主成分分析
    
'''
def deepPCA(X,y):
    pca = PCA(n_components=2)  ## 保留下来的特征数
    # 数据拟合
    pca.fit(X)
    #messagePrint(pca.explained_variance_ratio_)  # 0.92461872 0.05306648 0.01710261 0.00521218  选取前两个特征
    #messagePrint(pca.components_.T)
    #将原始矩阵 (150 x 4) 和转置主成分矩阵(4 * 2)相乘,得到投影数据 （150 x 2 )
    # 对原始数据进行z分数缩放
    x_scaled = StandardScaler().fit_transform(X)
    X1 = np.dot(x_scaled,pca.components_.T)[:5,]
    messagePrint(X1)

if __name__ == '__main__':
    pass