'''
特征工程相关操作
    1、数据基础分析
        a、空值分析
        b、分布分析
    2、
'''
from sklearn.model_selection import GridSearchCV

'''
获取模型评估指标
    输入参数：
        1、model:模型算法实例
        2、params : 需要验证匹配的超参数  tree_pipe_params = {'max_depth':[1,3,5,7]}，多个参数之间用逗号分隔
        3、样本数据集 X
        4、类别字段 Y
        
'''
def get_best_model_and_accuracy(model,params,X,y):
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
    print('  Best Accuracy : {}'.format(grid.best_score_))
    # 得到最佳准确率的最佳参数
    print( ' Best Parameters : {}'.format(grid.best_params_))
    # 拟合的平均时间（秒）
    print(" Average Time to Fit (s) ：{}".format(round(grid.cv_results_['mean_fit_time'].mean(),3)))
    # 预测的平均时间 （秒）,从该指标可以看出模型在真实世界的性能
    print(" Average Time to Score (s) : {}".format(round(grid.cv_results_['mean_score_time'].mean(),3)))


'''
获取数据集X的空值率
    输入：X - 样本数据集,特征名称，总字段数
    输出：每个样本的空值率的百分比
'''
def getFeatureNoneRate(X,features,all_sum):
    df1 = X.isnull().sum()  # 返回值为pandas.core.series.Series，并且以特征名称为索引
    for f in features:
        print(f + ' : ',round(df1[f]/all_sum,4) * 100)
    print('----------------------------------------------------')

'''
斯皮尔曼相关性系数
    斯皮尔曼相关系数表明X(独立变量)和Y(依赖变量)的相关方向。如果当X增加时，Y趋向于增加，
    斯皮尔曼相关系数则为正。如果当X增加时，Y趋向于减少，斯皮尔曼相关系数则为负。
    斯皮尔曼相关系数为零表明当X增加时Y没有任何趋向性
    
'''
def getSiPiEeMan(feature,x,y):
    # s.corr()函数计算
    r = round(abs(x.corr(y, method='spearman')),4)
    print(feature+":",r)