import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.dataMathHandle import getBins, cutFeautreData
from utils.handle_pyplot import showZft


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("D:/wo_yinyue_pex_xgxs.txt")  # 保存到D盘


le = LabelEncoder()
from time import strftime, localtime

# 打印当前时间
def printTime():
    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
    return

def handleFlagField(x):
    if x !=1 and x !=0:
        return 0
    else:
        return x
'''
    指定数据字段的类型及处理空值
'''

## 指标独热编码
def handeleOneHotCode(X):
    le.fit(X)
    return le.transform(X)

##  使用Z-SCORE方法进行归一化处理

def handleDataToOne(x):
    z_score = (x - x.mean()) / x.std()
    return abs(z_score)
'''
def handleDataToOne(x):
    return  x
'''

def handleAge(x):
    if x <=70 and x>=10:
        return x
    else:
        return 34

def changeType(data):
    data['prov_id'].fillna('099', inplace=True)
    data['prov_id'] = handeleOneHotCode(data['prov_id'])
    data['user_id'].fillna('9999999999', inplace=True)
    data['product_id'].fillna('9999999', inplace=True)
    data['area_id'].fillna('0991', inplace=True)
    data['device_number'].fillna('9999999999', inplace=True)
    data['cust_sex'].fillna('1', inplace=True)
    data['cert_age'] = data['cert_age'].apply(lambda x : handleAge(x))
    data["cert_age"] = data["cert_age"].round(2)
    data['total_fee'] = data['total_fee'].apply(pd.to_numeric, errors='coerce').fillna(60.66)
    data["total_fee"] = data["total_fee"].round(2)
    data['jf_flux'] = data['jf_flux'].apply(pd.to_numeric, errors='coerce').fillna(6.9)
    data["jf_flux"] = data["jf_flux"].round(1)
    data['fj_arpu'] = data['fj_arpu'].apply(pd.to_numeric, errors='coerce').fillna(5.4)
    data["fj_arpu"] = data["fj_arpu"].round(1)
    data['ct_voice_fee'] = data['ct_voice_fee'].apply(pd.to_numeric, errors='coerce').fillna(4.0)
    data["ct_voice_fee"] = data["ct_voice_fee"].round(1)
    data['total_flux'] = data['total_flux'].apply(pd.to_numeric, errors='coerce').fillna(16265.98)
    data["total_flux"] = data["total_flux"].round(2)
    data['total_dura'] = data['total_dura'].apply(pd.to_numeric, errors='coerce').fillna(322.45)
    data["total_dura"] = data["total_dura"].round(2)
    data['roam_dura'] = data['roam_dura'].apply(pd.to_numeric, errors='coerce').fillna(64.81)
    data["roam_dura"] = data["roam_dura"].round(2)
    data['total_times'] = data['total_times'].apply(pd.to_numeric, errors='coerce').fillna(434.88)
    data["total_times"] = data["total_times"].round(2)
    data['total_nums'] = data['total_nums'].apply(pd.to_numeric, errors='coerce').fillna(204.89)
    data["total_nums"] = data["total_nums"].round(2)
    data['local_nums'] = data['local_nums'].apply(pd.to_numeric, errors='coerce').fillna(156.85)
    data["local_nums"] = data["local_nums"].round(2)
    data['roam_nums'] = data['roam_nums'].apply(pd.to_numeric, errors='coerce').fillna(36.26)
    data["roam_nums"] = data["roam_nums"].round(2)
    data['in_cnt'] = data['in_cnt'].apply(pd.to_numeric, errors='coerce').fillna(100.04)
    data["in_cnt"] = data["in_cnt"].round(2)
    data['out_cnt'] = data['out_cnt'].apply(pd.to_numeric, errors='coerce').fillna(104.86)
    data["out_cnt"] = data["out_cnt"].round(2)
    data['in_dura'] = data['in_dura'].apply(pd.to_numeric, errors='coerce').fillna(152.31)
    data["in_dura"] = data["in_dura"].round(1)
    data['out_dura'] = data['out_dura'].apply(pd.to_numeric, errors='coerce').fillna(169.66)
    data["out_dura"] = data["out_dura"].round(1)
    data['heyue_flag'].fillna(0, inplace=True)
    data['is_limit_flag'].fillna('0', inplace=True)
    data['product_type'].fillna('other', inplace=True)
    data['product_type'] = handeleOneHotCode(data['product_type'])
    data['5g_flag'] = data['5g_flag'].apply(lambda x: handleFlagField(x))
    data['visit_cnt'] = data['visit_cnt'].apply(pd.to_numeric, errors='coerce').fillna(2231.73)
    data["visit_cnt"] = data["visit_cnt"].round(2)
    data['visit_dura'] = data['visit_dura'].apply(pd.to_numeric, errors='coerce').fillna(35562.76)
    data["visit_dura"] = data["visit_dura"].round(2)
    data['up_flow'] = data['up_flow'].apply(pd.to_numeric, errors='coerce').fillna(1.14)
    data["up_flow"] = data["up_flow"].round(2)
    data['down_flow'] = data['down_flow'].apply(pd.to_numeric, errors='coerce').fillna(4.67)
    data["down_flow"] = data["down_flow"].round(2)
    data['total_flow'] = data['total_flow'].apply(pd.to_numeric, errors='coerce').fillna(4.91)
    data["total_flow"] = data["total_flow"].round(2)
    data['active_days'] = data['active_days'].apply(pd.to_numeric, errors='coerce').fillna(9.96)
    data["active_days"] = data["active_days"].round(2)
    data['brand_flag'].fillna(13, inplace=True)
    data['imei_duration'] = data['imei_duration'].apply(pd.to_numeric, errors='coerce').fillna(12.0)
    data["imei_duration"] = data["imei_duration"].round(2)
    data['avg_duratioin'] = data['avg_duratioin'].apply(pd.to_numeric, errors='coerce').fillna(12.0)
    data["avg_duratioin"] = data["avg_duratioin"].round(2)
    #data['flag'].fillna(0, inplace=True)
    return data

printTime()
trainFilePath = 'D:/data/python/work/qwr_woyinyue_basic_result1.txt'
testFilePath = 'D:/data/python/work/qwr_woyinyue_user_result2006_87.txt'
## , 'fj_arpu'
all_params = ['prov_id','user_id','cust_id','product_id','area_id','device_number','cust_sex','cert_age','total_fee','jf_flux','fj_arpu',
              'ct_voice_fee','total_flux','total_dura','roam_dura','total_times','total_nums','local_nums','roam_nums','in_cnt','out_cnt',
              'in_dura','out_dura','heyue_flag','is_limit_flag','product_type','5g_flag','visit_cnt','visit_dura','up_flow','down_flow',
              'total_flow','active_days','brand','brand_flag','brand_detail','imei_duration','avg_duratioin']
labels = ['flag']
label = 'flag'

all_params1 = all_params
train = pd.read_csv(filepath_or_buffer=trainFilePath, sep="|", names=all_params + labels, encoding='utf-8')
test = pd.read_csv(filepath_or_buffer=testFilePath, sep="|", names=all_params, encoding='utf-8')

## 对空值和分类字段进行处理
train1 = changeType(train)
train1[label] = train1[label].apply(lambda x: handleFlagField(x))
test1 = changeType(test)

num_params = ['cert_age','total_fee','jf_flux','fj_arpu','ct_voice_fee','total_flux','total_dura','roam_dura','total_times',
              'total_nums','local_nums','roam_nums','in_cnt','out_cnt','in_dura','out_dura','visit_cnt','visit_dura','up_flow','down_flow',
              'total_flow','active_days','imei_duration','avg_duratioin']


'''
计算标准差
print(['样本数据：',train1.shape])
for columns1 in num_params:
    print(columns1 + ' :  ', train1[columns1].std())
'''

'''
print(['测试数据：',test1.shape])
for columns2 in num_params:
    print(columns2 + ' :  ', test1[columns2].std())
'''
## 生成DMatrix,字段内容必须为数字或者boolean    , 'fj_arpu'

from math import sqrt

def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab

def corrcoef(x, y):
    n = len(x)
    # 求和
    sum1 = sum(x)
    sum2 = sum(y)
    # 求乘积之和
    sumofxy = multipl(x, y)
    # 求平方和
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    # 计算皮尔逊相关系数
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    return num / den

'''
##  计算指标之间的相关性系数
    for columns1 in num_params:
    pex_xs = corrcoef(train1[columns1], test1[columns1])
    print(columns1 + '的皮尔逊相关性系数为 : ', pex_xs)
    
'''
## 特征分布直方图
#train1['cert_age'].hist(color='r', alpha=0.5, bins=30,xlabelsize=5)
# 不要加 plot，直接用 hist 即可多子图显示
# color 可以指定颜色
'''
feature = 'total_fee'
for feature in num_params:
    showZft(train1[feature],feature)
print(train1['cert_age'].max())
print(train1['cert_age'].min())
print(train1['cert_age'].mean())
1804.0
6.0
33.160949400000014
'''
#print(train1.query("cert_age > 70 or cert_age<10").shape)

## 切分数据，重新打标

'''
获取分段列表
feature1= feature+'1'
print(train1[feature].max())
print(train1[feature].min())
print(train1[feature].mean())
print(train1.query("total_fee<10").shape)
bins = getBins(0,13,1)
train1[feature1] = train1[feature].apply(lambda x : cutFeautreData(bins,x))
showZft(train1[feature1],feature1)

'''

'''
数据特征离散化
    cert_age : 不用特征工程
    total_fee:
        bins = getBins(0,200,3)
    jf_flux : 大部分数据等于零
    total_flux:
        bins = getBins(0,1024*40,512)
    imei_duration
        bins = getBins(0,36,1)
     in_cnt   
        bins = getBins(0,120,2)
    
df["author"].value_counts()  统计元素出现的次数

'''
feature = 'jf_flux'
print(train1[feature].max())
print(train1[feature].min())
print(train1[feature].mean())
#print(train1.query("jf_flux < 10").shape)
showZft(train1[feature],feature)

feature1= feature+'1'
bins = getBins(0,20,1)
train1[feature1] = train1[feature].apply(lambda x : cutFeautreData(bins,x))
showZft(train1[feature1],feature1)
