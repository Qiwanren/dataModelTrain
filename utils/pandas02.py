import pandas as pd
import numpy as np

## 列相关的操作
def method01():
    df1 = pd.DataFrame([['Snow', 'M', 22], ['Tyrion', 'M', 32], ['Sansa', 'F', 18], ['Arya', 'F', 14]],
                       columns=['name', 'gender', 'age'])

    print("----------在最后新增一列---------------")
    print("-------案例1----------")
    # 在数据框最后加上score一列，元素值分别为：80，98，67，90
    df1['score'] = [80, 98, 67, 90]  # 增加列的元素个数要跟原数据列的个数一样
    # print(df1)

    # df1.to_csv('D:/data/python/machine/sjsl_test.csv')

    ###  读取数据
    data = pd.read_csv("D:/data/python/machine/diabetes.csv",
                       usecols=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                                'DiabetesPedigreeFunction', 'Age', 'Outcome'])
    # data1 = data.iloc[:,3:8]
    # print(data1)

    # data2 = data['Outcome'] == 1
    # data2.info()
    data1 = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness']]
    data2 = data[['Pregnancies']]
    print(data1.info())
    print(data2.info())

## dataframe 与 array之间的转换
def method02():
    a = np.array([3, 1, 2, 4, 6, 1])
    print(type(a))
    data = pd.DataFrame(a)
    print(type(data))
    print(data)

def dataQuChong():
    data = pd.read_csv("D:/data/python/work/test.csv",names=['id', 'name', 'age'])
    print(data.head(10))
    data = data['id']  # 选择要去重的列
    data = set(data)  # 去重
    data = pd.DataFrame(list(data), columns=['item_id'])  # 因为set是无序的，必须要经过list处理后才能成为DataFrame
    data.to_csv('D:/data/python/work/test1.csv', index=False)  # 保存表格

    data1 = pd.read_csv("D:/data/python/work/test1.csv", names=['id', 'name', 'age'])
    print(data1.head())

##  条件查询字集
def queryData():
    data = pd.read_csv("D:/data/python/work/test.csv",names=['id', 'name', 'age'])
    ##print(data.head(10))
    data1 = data.query("id == '1001'")
    print(data1.head(10))

## 两个DataFrame相加
def dataFrameAdd():
    df1 = pd.DataFrame(np.random.randn(3, 5), columns=['a', 'b', 'c', 'd', 'e'])
    noise_df = pd.DataFrame(np.random.random(df1.shape), columns=df1.columns)
    print(df1)
    print(noise_df)
    data = df1 + noise_df
    print(data)

# 删除列
def dropLi():
    df1 = pd.DataFrame([['Snow', 'M', 22], ['Tyrion', 'M', 32], ['Sansa', 'F', 18], ['Arya', 'F', 14]],
                       columns=['name', 'gender', 'age'])
    df1 = df1.drop(['name', 'gender'], axis=1)  # axis=1 表示删除列，['密度', '含糖率'] 要删除的col的列表，可一次删除多列,df1不改变，改变显示
    print(df1)
    df1.drop(['name', 'gender'], axis=1, inplace=True)  # inplace=True, 直接从内部删除

def quchongxiaoshudian():
    n = 2.
    m = round(n)
    print(m)

    n = '023'
    s = n.zfill(3)
    print(s)
if __name__ == '__main__':
    quchongxiaoshudian()