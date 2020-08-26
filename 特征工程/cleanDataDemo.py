import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from utils.handle_pyplot import draw_heatmap, showZft2

'''

数据清洗

此次只处理定量数据
    1、删除数据
    2、使用均值等填充缺失数据
    
使用不同数据清洗方式，确定模型的准确率及泛化能力
    1、删除缺失值
    2、使用均值填充缺失值
    3、标准化，归一化
'''
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

def messagePrint(x):
    print(x)
    print('-------------------------------------------')

import matplotlib.pyplot as plt #  数据可视化工具
import seaborn as sns ## 另一个数据可视化工具
plt.style.use('fivethirtyeight')

# 引入第 3 个数据集(皮马印第安人糖尿病预测数据集)
pima_columns = ['times_pregment', 'plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_thickness',
                'serum_insulin', 'bmi', 'pedigree_function', 'age', 'onset_diabetes']

pima = pd.read_csv('D:/data/python/test/pima.data', names=pima_columns)
'''
空准确率：空准确率是指当模型总是预测频率较高类别时达到的正确率
'''
#messagePrint(pima['onset_diabetes'].value_counts(normalize=True))

'''
本模型的目标是预测是否会患糖尿病，因此对糖尿病患者和健康人的数据区分进行可视化
  通过图形得到：患者和正常人的血糖浓度有很大的差异
  
  
'''
col = 'plasma_glucose_concentration'
##  查看其它特征的数据差异.本次只处理定量数据
pima_columns1 = ['plasma_glucose_concentration', 'diastolic_blood_pressure', 'bmi']
for col in pima_columns1:
    plt.hist(pima[pima['onset_diabetes'] == 0][col], 10, alpha=0.5, label='non-diabetes')
    plt.hist(pima[pima['onset_diabetes'] == 1][col], 10, alpha=0.5, label='diabetes')
    plt.legend(loc='upper right')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title('Histogram of {}'.format(col))
    #plt.show()

# 数据集相关矩阵的热力图
pima1 = pima.copy(deep=True)
#new_col = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9']
#pima1.columns = new_col
#draw_heatmap(pima1)

#print(pima.isnull().sum())
#print(pima.describe())  ## 根据结果中显示：bmi列最小值为 0 ，不符合实际数据

'''
处理缺失值
    1、删除缺失行
    2、填补缺失数据
    
字段含义介绍：
    怀孕次数,口服葡萄糖耐量试验中血浆葡萄糖浓度,舒张压（mm Hg）,三头肌组织褶厚度（mm）,2小时血清胰岛素（μU/ ml） 
    体重指数（kg/（身高(m)）^ 2）,糖尿病系统功能,年龄（岁） 
'''
## 先用NULL填充数据中的0字段
## 因为怀孕次数可以为零，因此times_pregment的最小值为零是合理的，因此不用处理
zero_feature = ['f2', 'f3', 'f4','f5', 'f6']
zero_feature = ['plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_thickness','serum_insulin', 'bmi']
'''
将dataFrame中的零值替换为None
'''
for f1 in zero_feature:
    #pima[f1] = pima[f1].map(lambda x:x if x !=0 else None)
    pima[f1].replace([0],[None],inplace=True)
#print(pima.isnull().sum())
#print(pima.head())

# 移除缺失值
pima_dropped = pima.isnull().sum()
num = pima.shape[0]
# messagePrint(data_dropped)
#pima.dropna(axis=0, inplace=True)

## 特征变化比
# print(train1_sub_china.shape[0])
fe_drop_data_index = pima_dropped.index.tolist()
fe_drop_data_values = pima_dropped.values
f1 = np.arange(0, len(fe_drop_data_index), 1)
'''
for i in f1:
   print(fe_drop_data_index[i] + ' : ', round(fe_drop_data_values[i] / num, 2))
'''

##messagePrint(pima.shape)

'''
通过比较pima_dropped，pima,以及删除数据后的pima，发现数据发生了很大的变化，因此一般不建议删除数据
比较维度：
    1、空准确率：空准确率是指当模型总是预测频率较高类别时达到的正确率
    2、数据平均值
    
'''
# 均值的变化 , pimal1为原始值，pima为修改后的值
'''
tirnes_pregnant (怀孕次数)的均值在删除缺失值后下降了 14% ，变化很大!
pedigree_function (糖尿病血系功能)也上升了 11% ，也是个飞跃 可以看到，删除行(观察值)会严重影响数据的形状，
'''
#a1 = (pima.mean() - pima1.mean())/pima1.mean()
#print(a1)
#----------+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++--------------#
'''
填充缺失值 ： 填充指的是利用现有知识/数据来确定缺失的数量值并填充的行为
    1、使用剩余数据的均值进行填充
    2、
字段含义介绍：
    怀孕次数,口服葡萄糖耐量试验中血浆葡萄糖浓度,舒张压（mm Hg）,三头肌组织褶厚度（mm）,2小时血清胰岛素（μU/ ml） 
    体重指数（kg/（身高(m)）^ 2）,糖尿病系统功能,年龄（岁） 
    
'''
# 使用均值填充缺失值
pima_columns1 = ['times_pregment', 'plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_thickness',
                'serum_insulin', 'bmi', 'pedigree_function', 'age']
### 第一种方法，填充平均值
'''
    for f in pima_columns1:
    avg_n = round(pima[f].mean(),2)
    pima[f] = pima[f].map(lambda x:x if x!=None else avg_n)
    a1 = (pima.mean() - pima1.mean())/pima1.mean()
    
'''
#messagePrint(a1)

### 第二种方法，填充平均值
imputer = SimpleImputer(strategy='mean')
pima_imputed = imputer.fit_transform(pima)
pima = pd.DataFrame(pima_imputed)
pima.columns = pima_columns

##  查看平均值填充的效果
a1 = (pima.mean() - pima1.mean())/pima1.mean()
messagePrint(a1)
