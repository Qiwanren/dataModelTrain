import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils.handle_pyplot import showZft

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
'''
目录：
    方法1：基础操作，处理空值
    方法2：处理分类数据




'''

'''
    特征基础操作：
        1、数据分布
        2、空值检查和处理
'''


def messagePrint(x):
    print(x)
    print('-------------------------------------------')

def method01():
    path1 = 'D:/data/python/test/GlobalLandTemperaturesByCity.csv'
    # path2 = 'D:/data/python/test/Salary_Ranges_by_Job_Classification.csv'

    train1 = pd.read_csv(filepath_or_buffer=path1, encoding='utf-8')
    # train2 = pd.read_csv(filepath_or_buffer=path2, encoding='utf-8')

    # print(train1.shape)

    ### 第二步，简单的字段处理（日期，变量类别转换）
    # 日期转换, 将dt 转换为日期，取年份, 注意map的用法
    train1['dt'] = pd.to_datetime(train1['dt'])
    train1['year'] = train1['dt'].map(lambda value: value.year)

    # 只看中国
    train1_sub_china = pd.DataFrame(train1.loc[train1['Country'] == 'China'])
    # 通过年份数据，增加世纪字段
    train1_sub_china['Century'] = train1_sub_china['year'].map(lambda x: int(x / 100 + 1))

    # print(train1_sub_china[['year','Century']])
    # print(train1_sub_china.head())

    '''
    特征增强：
        1、首先看看目标占比情况（针对二分类问题，也就是0和1的占比情况），直接 value_counts()就可以解决，看看样本是否失衡
        2、查看是否有空值，直接统计 isnull().sum() 的个数，不过需要注意的是，可能统计出来没有缺失，并不是因为真的没有缺失，而且缺失被
           人用某个特殊值填充了，一般会用 -9、blank、unknown、0之类的，需要注意⚠️识别，后面需要对缺失进行合理填充，通过 data.describe() 
           获取基本的描述性统计，根据均值、标准差、极大极小值等指标，结合变量含义来判断

           常用检查缺失值的方法：删除和填充
                1）
    '''
    # print(train1_sub_china.isnull().sum())
    # print(train1_sub_china.describe())
    ###   --+++++++++++++++++++++++++++++++++++++++++++++++++++++第一步，处理空值或者异常值 +++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 处理被错误填充的缺失值0，还原为 空(单独处理)
    features = ['AverageTemperature', 'AverageTemperatureUncertainty']
    for feature in features:
        train1_sub_china[feature] = train1_sub_china[feature].map(lambda x: x if x != 0 else None)  ## 如果x != 0，则返回原来值，如果 x == 0,则设置为空
        # print('AverageTemperature null :',train1_sub_china['AverageTemperature'].isnull().sum())
    # 检查变量缺失情况
    # print(train1_sub_china.isnull().sum())

    # 移除缺失值
    data_dropped = train1_sub_china.isnull().sum()
    # messagePrint(data_dropped)
    train1_sub_china.dropna(axis=0, inplace=True)
    # 检查变量缺失情况
    # messagePrint(train1_sub_china.isnull().sum())
    ## 查看数据规模
    # print(train1_sub_china.shape)

    ## 特征变化比
    # print(train1_sub_china.shape[0])
    fe_drop_data_index = data_dropped.index.tolist()
    fe_drop_data_values = data_dropped.values
    f1 = np.arange(0, len(fe_drop_data_index), 1)
    for i in f1:
        print(fe_drop_data_index[i] + ' : ', round(fe_drop_data_values[i] / 827802, 2))

'''
特征增强
    1、处理分类数据
        对分类变量进行填充操作，类别变量一般用众数或者特殊值来填充
'''
def method02():
    # 引入第 3 个数据集(皮马印第安人糖尿病预测数据集)
    pima_columns = ['times_pregment', 'plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_thickness',
                    'serum_insulin', 'bmi', 'pedigree_function', 'age', 'onset_disbetes']

    pima = pd.read_csv('D:/data/python/test/pima.data', names=pima_columns)
    #print(pima.head())
    #print(pima.info())
    #print(pima.describe())
    ## 再接着看不同类别之间的特征值分布情况，可通过画直方图（数值型变量）和计算变量值占比分布（类别变量）来观察

    # 处理被错误填充的缺失值0，还原为 空(单独处理)
    pima['serum_insulin'] = pima['serum_insulin'].map(lambda x: x if x != 0 else None)
    # 检查变量缺失情况
    #messagePrint(pima['serum_insulin'].isnull().sum())
    # 批量操作 还原缺失值
    columns = ['serum_insulin', 'bmi', 'plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_thickness']
    for col in columns:
        pima[col].replace([0], [None], inplace=True)

    # 检查变量缺失情况
    #messagePrint(pima.isnull().sum())

    # 删除含有缺失值的行
    pima_dropped = pima.dropna()
    '''
        num_rows_lost = round(100 * ((pima.shape[0] - pima_dropped.shape[0]) / pima.shape[0]))
        messagePrint("保留了原先 {}% 的行".format(num_rows_lost))
    
        # 查看下 删除行 之后，各个特征均值的差异
        plt.figure()
        ax = (100 * (pima_dropped.mean() - pima.mean()) / pima.mean()).plot(kind='bar', title='各个特征均值的改变%',)
        ax.set_ylabel('% change')
        plt.show()
    '''

    # 使用sklearn的 Pipeline以及 Imputer来实现缺失值填充
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.impute import SimpleImputer

    # 调参候选
    knn_params = {'classify__n_neighbors': [1, 2, 3, 4, 5, 6]}

    # 实例化KNN模型
    knn = KNeighborsClassifier()

    # 管道设计
    mean_impute = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                            ('classify', knn)
                            ])

    x = pima.drop('onset_disbetes', axis=1)  # 丢弃y
    y = pima['onset_disbetes']

    # 网格搜索
    grid = GridSearchCV(mean_impute, knn_params)
    grid.fit(x, y)

    # 打印模型效果
    print(grid.best_score_, grid.best_params_)

def method03():
    import matplotlib.pyplot as plt #  数据可视化工具
    import seaborn as sns ## 另一个数据可视化工具
    plt.style.use('fivethirtyeight')

    # 引入第 3 个数据集(皮马印第安人糖尿病预测数据集)
    pima_columns = ['times_pregment', 'plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_thickness',
                    'serum_insulin', 'bmi', 'pedigree_function', 'age', 'onset_diabetes']

    pima = pd.read_csv('D:/data/python/test/pima.data', names=pima_columns)
    # 空准确率
    messagePrint(pima['onset_diabetes'].value_counts(normalize=True))

if __name__ == '__main__':
    method03()

