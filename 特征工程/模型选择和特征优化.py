from copy import deepcopy

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier as RFC


from 特征工程.checkFeature.CustomCorrelationChooser import CustomCorrelationChooser

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
'''
特征选择主要是通过两方面进行优化
    1、根据特征相关性进行特征选择
        a、F检验，又称ANOVA，方差齐性检验
        b、卡方检验
        c、互信息
        
    2、利用现有模型对特征进行选择
        a、基于决策树的特征选择
        b、基于线性模型的特征选择
    
'''
def messagePrint(x):
    print(x)
    print('----------------------------------------')
# ------------------------------------------------------------基础部分--------------------------------------------------------------------
## 返回数据集
def method01():
    np.random.seed(123)
    path = 'D:/data/python/test/credit_card_default.csv'

    # 导入数据集
    credit_card_default = pd.read_csv(path)
    # 进行基础数据的信息分析
    messagePrint(credit_card_default.shape)
    #messagePrint(credit_card_default.describe().T)
    # 特征矩阵
    X = credit_card_default.drop('default payment next month',axis=1)
    # 响应变量
    y = credit_card_default['default payment next month']
    # 输出空准确率
    #print(y.value_counts(normalize=True))
    return credit_card_default,X,y

'''
选择最优模型
    1、逻辑回归
    2、K最近邻（KNN）
    3、决策树
    4、随机森林
    
'''
'''
输入参数为模型名称，模型实例，分析参数值，X，y数据域
输出模型最优准确率，拟合时间等相关参数
'''
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

# 输入参数为一个dataFrame，显示数据热力图
def showheatMap(df):
    import seaborn as sns
    import matplotlib.style as style
    # 选用一个干净的主题
    style.use('fivethirtyeight')
    sns.heatmap(df.corr())
    plt.show()
'''
模型算法选择
'''
def initModelCheck():
    # 逻辑回归
    lr_params = {'max_iter': [20, 40, 60, 100, 120],'intercept_scaling': [1],'C':[0.01, 0.1, 1, 10]}
    # KNN
    knn_params = {'n_neighbors': [1, 3, 5, 7]}
    # 决策树
    tree_params = {'max_depth': [None, 1, 3, 5, 7]}
    # 随机森林
    forest_params = {'n_estimators': [10, 50, 100], 'max_depth': [None, 1, 3, 5, 7]}

    # 实例化模型
    lr = LogisticRegression()
    knn = KNeighborsClassifier()
    d_tree = DecisionTreeClassifier()
    forest = RandomForestClassifier()
    credit_card_default,x,y = method01()
    # 预测模型
    get_best_model_and_accuracy('LogisticRegression',lr,lr_params,x,y)
    get_best_model_and_accuracy('KNeighborsClassifier', knn, knn_params, x, y)
    get_best_model_and_accuracy('DecisionTreeClassifier', d_tree, tree_params, x, y)
    get_best_model_and_accuracy('RandomForestClassifier', forest, forest_params, x, y)
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------根据列相关性选择特征列--------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------
'''
列相关性选择 ：通过列相关性（corr）选择在某个模型中组合最优的列
    方法：1、F检验，又称ANOVA，方差齐性检验
          2、互信息法 
          3、卡方检验
      注意： 以上三个计算维度都是统计特征与标签之间的关系
          4、自由度
    F检验-详解：
       1）F检验的本质是寻找两组数据之间的线性关系，其原假设是”数据不存在显著的线性关系“。它返回F值和p值两个统计量。和卡方过滤一样，
          希望选取p值小于0.05或0.01的特征，这些特征与标签时显著线性相关的，而p值大于0.05或0.01的特征则被我们认为是和标签没有显著线性关系的特征，
          应该被删除
       2）用来捕捉每个特征与标签之间的线性关系的过滤方法。即可以做回归也可以做分类，因此包含feature_selection.f_classif（F检验分类） 
          和 feature_selection.f_regression（F检验回归）两个类。其中F检验分类用于标签是离散型变量的数据，而F检验回归用于标签是连续型变量的数据。
       3）需要和类SelectKBest连用，可以直接通过输出的统计量来判断我们到底要设置一个什么样的K，需要注意的是，F检验在数据服从正态分布时效果会非常稳定，
          因此如果使用F检验过滤，我们会先将数据转换成服从正态分布的方式
    互信息详解
        1）互信息(Mutual Information)是信息论里一种有用的信息度量，它可以看成是一个随机变量中包含的关于另一个随机变量的信息量，1表示完全关联，0表示没有联系。
           结果值是[0,1]之间的概率值
        2）分为互信息分类和互信息回归
        
    卡方检验详解：
        1）卡方过滤是专门针对离散型标签（即分类问题）的相关性过滤
        2）卡方检验类计算每个非负特征和标签之间的卡方统计量，并依照卡方统计量由高到底为特征排名
        3）结合feature_selection.SelectKBest这个可以输入"评分标准"来选出前K个分数最高的特征的类
       作用：可以依据此除去最可能独立，与我们分类目的无关的特征

'''

'''
F值检验，卡方检验
    原理：假设数据特征之间不存在显著的线性关系，从而求得该假设为真的概率
    输入值： features_df 为特征数据集
             label_df  为结果标签数据集
             
    限制：需将数据转换成服从正态分布
    实现方式：
        from sklearn.feature_selection import f_classif     # 用于分类特征分析
        from sklearn.feature_selection import f_regression  # 用于连续特征分析
        from sklearn.feature_selection import chi2          # 卡方检验
        # 只保留最佳的五个特征，
        k_best = SelectKBest(f_classif,k=5)
        k_best = SelectKBest(f_regression,k=5)
        k_best = SelectKBest(chi2,k=5)  ## 选取相关性最高的5个特征
    
    结果：F值检验和卡方检验在输出结果的时候都会输出一个P值，该P值的意义:假设特征与响应变量之间没有关系的概率，因此p值越低，则特征与响应变量的关系越大
        
'''
def chooseFeatureBy_P_value(X,y):
    #selectKBest在给定目标函数后选择k个最高分
    #from sklearn.feature_selection import SelectKBest
    # ANOVA测试
    from sklearn.feature_selection import f_classif     # 用于分类特征分析
    from sklearn.feature_selection import f_regression  # 用于连续特征分析
    from sklearn.feature_selection import chi2          # 卡方检验
    # 只保留最佳的五个特征，
    k_best = SelectKBest(f_classif,k=5)
    #k_best = SelectKBest(f_regression,k=5)
    #k_best = SelectKBest(chi2,k=5)

    # 选择最佳特征的矩阵
    df1 = k_best.fit_transform(X,y)  # 获取选定的结果数据集

    # 获取列的p值
    messagePrint(k_best.pvalues_)
    #特征和p值组成DataFrame
    #按p值排列
    p_values = pd.DataFrame({'column':X.columns,'p_value':k_best.pvalues_}).sort_values('p_value')
    #选择阈值为0.05的特征
    p_values = p_values[p_values['p_value'] < 0.05]
    # 取前五个特征
    messagePrint(p_values)

'''
设置超参数选取chooseFeatureBy_P_value方法中的K值
    输入参数：模型对象，计算指标（卡方，F值），X:特征数据集，y:标签数据集
    根据显示图像确定表现最佳的k值
'''
def getKBestValue(model,count_rule,X,y):
    model = RFC(n_estimators=10,random_state=0)
    score = []
    for i in range(390,200,-10):
        x_fschi = SelectKBest(count_rule,k = i).fit_transform(X,y)
        # 验证在模型中的效果
        once = cross_val_score(model,x_fschi,y,cv=5).mean()
        score.append(once)
    plt.plot(range(390,200,-10),score)
    plt.show()

'''
互信息：
    mutual_info_classif : 互信息分类，可以捕捉任何相关性，不能用于稀疏矩阵
    mutual_info_regression : 互信息回归，可以捕捉任何相关性，不能用于稀疏矩阵

discrete_features：{'auto', bool, array_like}, default ‘auto’
    如果为'auto'，则将其分配给False（表示稠密）X，将其分配给True（表示稀疏）X。
    如果是bool，则确定是考虑所有特征是离散特征还是连续特征。
    如果是数组，则它应该是具有形状（n_features，）的布尔蒙版或具有离散特征索引的数组。
n_neighbors: int, default=3
    用于连续变量的MI估计的邻居数;较高的值会减少估计的方差，但可能会带来偏差。
copy: bool, default=True
    是否复制给定的数据。如果设置为False，则初始数据将被覆盖。
random_state: int, RandomState instance or None, optional, default None
    确定随机数生成，以将小噪声添加到连续变量中以删除重复值。
    在多个函数调用之间传递int以获得可重复的结果。
Returns
    mi: ndarray, shape (n_features,) 每个功能和目标之间的估计相互信息。

'''
def huxingxi(X,y):
    #from functools import partial
    #k_best = SelectKBest(score_func=partial(mutual_info_classif, random_state=0))

    # 选择最佳特征的矩阵
    #df1 = k_best.fit_transform(X, y)  # 获取选定的结果数据集
    minfo = mutual_info_classif(X, y, discrete_features=False, n_neighbors=3)
    print(minfo)


'''         
输入参数为：
    模型算法名称,模型对象，模型参数，数据域 X，y
原理：皮尔逊相关系数，同时皮尔逊系数要求每列数据正态分布的时候，表现最优
'''
def chooseFeatureByCorr(modelName,model,params,X,Y):
    from copy import deepcopy
    # 使用响应变量初始化热证选择器
    ccc = CustomCorrelationChooser(threshold=0.1,response=y)
    ccc.fit(X)
    messagePrint(ccc.cols_to_keep)
    # 创建流水线，包括选择器
    ccc_pipe = Pipeline([('correlation_select',ccc),
                         ('classifier',model)])
    # 复制决策树的参数
    #ccc_pipe_params = deepcopy(tree_pipe_params)
    ccc_pipe_params = deepcopy(params)
    # 更新决策树的参数选择
    ccc_pipe_params.update({'correlation_select__threshold':[0,0.1,0.2,0.3]})

    messagePrint(ccc_pipe_params)
    # 模型效果
    get_best_model_and_accuracy(modelName,ccc_pipe,ccc_pipe_params,X,Y)


# --------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------基于模型的特征选择--------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------

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

if __name__ == '__main__':
    # 模型选择
    #initModelCheck()
    credit_card_default, x, y = method01()
    #showheatMap(credit_card_default)

    # 特征选择,基于corr
    #d_tree = DecisionTreeClassifier()
    #tree_pipe_params = {'classifier__max_depth': [None, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
    #chooseFeatureByCorr('DecisionTreeClassifier',d_tree,tree_pipe_params,x,y)

    #基于P值的特征选择
    #chooseFeatureBy_P_value(x,y)

    # 特征选择,基于决策树
    d_tree = DecisionTreeClassifier()
    #chooseFeatureByDecisionTree(d_tree,x,y)

    # 线性模型和正则化
    tree = DecisionTreeClassifier()
    zehAndRegression(tree,x,y)
