import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
### 处理类别型特征
def one_hot(dataFrame,feature):
    dummies = pd.get_dummies(dataFrame[feature], prefix=feature)
    print(dummies)
    return dataFrame

## 删除完成one-hot编码的特征
def dropFeature(dataF,features):
    dataF.drop(features, axis=1, inplace=True)  # inplace=True, 直接从内部删除

def changeType(data):
    data['user_id'].fillna('9999999999', inplace=True)
    data['product_id'].fillna('9999999', inplace=True)
    data['area_id'].fillna('0991', inplace=True)
    data['device_number'].fillna('9999999999', inplace=True)
    data['cust_sex'].fillna('1', inplace=True)
    data['cert_age'] = data['cert_age'].apply(pd.to_numeric, errors='coerce').fillna(34.98)
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
    data['imei_duration'] = data['imei_duration'].apply(pd.to_numeric, errors='coerce').fillna(12.0)
    data["imei_duration"] = data["imei_duration"].round(2)
    data['avg_duratioin'] = data['avg_duratioin'].apply(pd.to_numeric, errors='coerce').fillna(12.0)
    data["avg_duratioin"] = data["avg_duratioin"].round(2)
    #data['flag'].fillna(0, inplace=True)
    return data

def handleTypeFlag(n):
    return round(n)

def handleBrandFlag(data):
    data['brand_flag'] = data['brand_flag'].apply(pd.to_numeric, errors='coerce').fillna(13)
    data['cust_sex'].fillna(1, inplace=True)
    return data




printTime()
trainFilePath = 'D:/data/python/work/qwr_woyinyue_basic_result1.txt'
testFilePath = 'D:/data/python/work/qwr_woyinyue_user_result2006_087.txt'

all_params = ['prov_id','user_id','cust_id','product_id','area_id','device_number','cust_sex','cert_age','total_fee','jf_flux','fj_arpu',
              'ct_voice_fee','total_flux','total_dura','roam_dura','total_times','total_nums','local_nums','roam_nums','in_cnt','out_cnt',
              'in_dura','out_dura','heyue_flag','is_limit_flag','product_type','5g_flag','visit_cnt','visit_dura','up_flow','down_flow',
              'total_flow','active_days','brand','brand_flag','brand_detail','imei_duration','avg_duratioin']

labels = ['flag']
label = 'flag'

all_params1 = all_params
train = pd.read_csv(filepath_or_buffer=trainFilePath, sep="|", names=all_params + labels, encoding='utf-8')
test = pd.read_csv(filepath_or_buffer=testFilePath, sep=",", names=all_params, encoding='utf-8')


train = handleBrandFlag(train)
test = handleBrandFlag(test)

train['cust_sex'] = train['cust_sex'].apply(lambda x : handleTypeFlag(x))
test['cust_sex'] = test['cust_sex'].apply(lambda x : handleTypeFlag(x))

print(train['cust_sex'].unique())
print(test['cust_sex'].unique())
## 对分类特征进行one-hot编码
train1 = one_hot(train, 'cust_sex')
test1 = one_hot(test,'cust_sex')

