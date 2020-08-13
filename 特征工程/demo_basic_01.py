import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

def messagePrint(x):
    print(x)
    print('-------------------------------------------')


path1 = 'D:/data/python/test/GlobalLandTemperaturesByCity.csv'
#path2 = 'D:/data/python/test/Salary_Ranges_by_Job_Classification.csv'

train1 = pd.read_csv(filepath_or_buffer=path1, encoding='utf-8')
#train2 = pd.read_csv(filepath_or_buffer=path2, encoding='utf-8')

#print(train1.shape)

### 第二步，简单的字段处理（日期，变量类别转换）
# 日期转换, 将dt 转换为日期，取年份, 注意map的用法
train1['dt'] = pd.to_datetime(train1['dt'])
train1['year'] = train1['dt'].map(lambda value: value.year)

# 只看中国
train1_sub_china = pd.DataFrame(train1.loc[train1['Country'] == 'China'])
# 通过年份数据，增加世纪字段
train1_sub_china['Century'] = train1_sub_china['year'].map(lambda x: int(x / 100 + 1))


#print(train1_sub_china[['year','Century']])
#print(train1_sub_china.head())

'''
特征增强：
    1、首先看看目标占比情况（针对二分类问题，也就是0和1的占比情况），直接 value_counts()就可以解决，看看样本是否失衡
    2、查看是否有空值，直接统计 isnull().sum() 的个数，不过需要注意的是，可能统计出来没有缺失，并不是因为真的没有缺失，而且缺失被
       人用某个特殊值填充了，一般会用 -9、blank、unknown、0之类的，需要注意⚠️识别，后面需要对缺失进行合理填充，通过 data.describe() 
       获取基本的描述性统计，根据均值、标准差、极大极小值等指标，结合变量含义来判断
       
       常用检查缺失值的方法：删除和填充
            1）
'''
#print(train1_sub_china.isnull().sum())
#print(train1_sub_china.describe())
###   --+++++++++++++++++++++++++++++++++++++++++++++++++++++第一步，处理空值或者异常值 +++++++++++++++++++++++++++++++++++++++++++++++++++++
# 处理被错误填充的缺失值0，还原为 空(单独处理)
features = ['AverageTemperature','AverageTemperatureUncertainty']
for feature in features:
    train1_sub_china[feature] = train1_sub_china[feature].map(lambda x: x if x != 0 else None)    ## 如果x != 0，则返回原来值，如果 x == 0,则设置为空
    #print('AverageTemperature null :',train1_sub_china['AverageTemperature'].isnull().sum())
# 检查变量缺失情况
#print(train1_sub_china.isnull().sum())

# 批量操作 还原缺失值
'''
columns = ['serum_insulin', 'bmi', 'plasma_glucose_concentration', 'diastolic_blood_pressure', 'triceps_thickness']
for col in columns:
    train1_sub_china[col].replace([0], [None], inplace=True)
'''

# 移除缺失值
data_dropped = train1_sub_china.isnull().sum()
#messagePrint(data_dropped)
train1_sub_china.dropna(axis=0, inplace=True)
# 检查变量缺失情况
#messagePrint(train1_sub_china.isnull().sum())
## 查看数据规模
#print(train1_sub_china.shape)

## 特征变化比
#print(train1_sub_china.shape[0])
messagePrint(type(data_dropped))
messagePrint(data_dropped.index.tolist())
messagePrint(data_dropped.values)
fe_drop_data_index = data_dropped.index.tolist()
fe_drop_data_values = data_dropped.values
f1 = np.arange(0, len(fe_drop_data_index), 1)
for i in f1:
    print(fe_drop_data_index[i] + ' : ',round(fe_drop_data_values[i]/827802),2)
