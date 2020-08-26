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
    a = np.array([3, 1, 2, 4, 6, 1],[3, 1, 2, 4, 6, 1])
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

def twoListToDataFrame():
    name = ['Cindy', 'John', 'Matt']
    point = [78, 87, 88]
    df_grade = pd.DataFrame(name, columns=['name'])
    df_grade = pd.concat([df_grade, pd.DataFrame(point, columns=['point'])], axis=1)
    print(df_grade)


from sklearn import preprocessing
import pandas as pd

def min_max_value():
    df1 = pd.DataFrame([['Snow', 'M', 22], ['Tyrion', 'M', 32], ['Sansa', 'F', 18], ['Arya', 'F', 14]],
                       columns=['name', 'gender', 'age'])
    print(df1.info())
    min_max_scaler = preprocessing.MinMaxScaler()
    x_minmax = min_max_scaler.fit_transform(df1['age'])
    print(x_minmax)

def method055():
    type_feature = ['prov_id', 'cust_sex', 'brand_flag', 'heyue_flag', 'is_limit_flag', 'product_type', '5g_flag',
                    'flag']

    type_feature.remove('flag')
    print(type_feature)
def renameFeature(df):
    print(df.info())
    list_cols = df.columns.values
    #print(list_cols.size)
    arr1 = np.arange(0,list_cols.size,1)
    #print(arr1)
    df.columns = arr1
    print(df.info())

'''
根据列的值删除行
'''
def method01():
    data = {'city': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Hangzhou', 'Chongqing'],
            'year': [2016, 2016, 2015, 2017, 2016, 2016],
            'population': [2100, 2300, 1000, 700, 500, 500]}
    frame = pd.DataFrame(data, columns=['year', 'city', 'population', 'debt'])
    condition = ['Beijing', 'Shanghai','Guangzhou']
    print(frame)
    print("————————————删除列 ————————————")
    df = frame.drop(frame[~frame.city.isin(condition)].index)
    print(df)

def jiequzifuchuan():
    test_path1 = 'D:/data/python/work/qwr_woyinyue_user_result2006_087.txt'
    test_path2 = 'D:/data/python/work/qwr_woyinyue_user_result2006_036.txt'
    paths = [test_path1, test_path2]
    for str in paths:
        print(str)
        filename = str[-7:-4]
        path = 'D:/data/python/work/result_' + filename + '.csv'
        print(path)

def twoDf_pingjie(df1,df2):
    pass

if __name__ == '__main__':
    #method055()
    data1 = pd.read_csv("D:/data/python/test/credit_card_default.csv")
    #method01(data1)
    jiequzifuchuan()