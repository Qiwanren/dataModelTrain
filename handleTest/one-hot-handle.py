import pandas as pd


def one_hot(dataFrame,feature):
    dummies = pd.get_dummies(dataFrame[feature], prefix='gender')
    print(dummies)
    df = pd.concat([dataFrame, dummies], axis=1)
    print(df)


labels = ['flag']
all_params = ['prov_id', 'user_id', 'cust_id', 'product_id', 'area_id', 'device_number', 'cust_sex', 'cert_age',
              'total_fee', 'jf_flux', 'fj_arpu',
              'ct_voice_fee', 'total_flux', 'total_dura', 'roam_dura', 'total_times', 'total_nums', 'local_nums',
              'roam_nums', 'in_cnt', 'out_cnt',
              'in_dura', 'out_dura', 'heyue_flag', 'is_limit_flag', 'product_type', '5g_flag', 'visit_cnt',
              'visit_dura', 'up_flow', 'down_flow',
              'total_flow', 'active_days', 'brand', 'brand_flag', 'brand_detail', 'imei_duration', 'avg_duratioin']

type_feature = ['prov_id','brand_flag','heyue_flag','is_limit_flag','product_type','5g_flag']

trainFilePath = 'D:/data/python/work/qwr_woyinyue_basic_result3.txt'
train = pd.read_csv(filepath_or_buffer=trainFilePath,header=None,sep="|", names=all_params + labels, encoding='utf-8')


one_hot(train,'product_type')