import pandas as pd

def loadDataAndWriteData(path,filename):
    all_params = ['prov_id', 'user_id', 'cust_id', 'product_id', 'area_id', 'device_number', 'cust_sex', 'cert_age',
                  'total_fee', 'jf_flux', 'fj_arpu',
                  'ct_voice_fee', 'total_flux', 'total_dura', 'roam_dura', 'total_times', 'total_nums', 'local_nums',
                  'roam_nums', 'in_cnt', 'out_cnt',
                  'in_dura', 'out_dura', 'heyue_flag', 'is_limit_flag', 'product_type', '5g_flag', 'visit_cnt',
                  'visit_dura', 'up_flow', 'down_flow',
                  'total_flow', 'active_days', 'brand', 'brand_flag', 'brand_detail', 'imei_duration', 'avg_duratioin',
                  'service_type']

    labels = ['flag']

    data = pd.read_csv(filepath_or_buffer=path, sep="|", names=all_params + labels, encoding='utf-8')

    df1 = data.query("service_type in ('40AAAAAA','50AAAAAA','90AAAAAA') ")

    print(df1['service_type'].value_counts())

    df1.to_csv('D:/data/python/work/data1/' + filename,index=None)

if __name__ == '__main__':
    path1 = 'D:/data/python/work/qwr_woyinyue_basic_result0501.txt'
    path2 = 'D:/data/python/work/qwr_woyinyue_basic_result0502.txt'
    path3 = 'D:/data/python/work/qwr_woyinyue_basic_result0503.txt'
    path4 = 'D:/data/python/work/qwr_woyinyue_basic_result0504.txt'

    loadDataAndWriteData(path1, 'qwr_woyinyue_basic_result0501.txt')
    loadDataAndWriteData(path2, 'qwr_woyinyue_basic_result0502.txt')
    loadDataAndWriteData(path3, 'qwr_woyinyue_basic_result0503.txt')
    loadDataAndWriteData(path4, 'qwr_woyinyue_basic_result0504.txt')