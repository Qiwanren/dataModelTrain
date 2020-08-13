# 训练模型


# 查看数据维度
print('数据的维度为', data.shape)
n = data.shape[0]
m = data.shape[1]

# 剔除掉  无用字段
remove_col = list(
    ['phone', 'k000077', 'k000078', 'k000079', 'k000080', 'k000087', 'k000088', 'k002037', 'k002039', 'k002062',
     'k002085', 'k002108', 'k002191', 'k002195', 'k002204',
     'k003667', 'k003679', 'k003680', 'k003748', 'k003749', 'k003750', 'k003753', 'k003754', 'k003755', 'k003758',
     'k003759', 'k003760', 'k003761', 'k003762', 'k003763', 'k003764',
     'k003765', 'k003766', 'k003767', 'k003768', 'k003775', 'k003778', 'k003779', 'k003789', 'k003791', 'k003792',
     'k003793', 'k003794', 'k003795', 'k003802', 'k003803', 'k003804',
     'k003805', 'k003806', 'k003853', 'k003854', 'k003868', 'k003869', 'k003870', 'k003874', 'k003875', 'k003884',
     'k003885', 'k003887', 'k000058', 'k003773', 'k002052', 'k003796'])

data.drop(remove_col, inplace=True, axis=1)
print(data.shape)

# 对业务数据指标进行分组 1、基础信息指标  2、业务信息指标  3、终端信息指标  4、业务使用行为指标

# 1、基础信息指标
# k000001 - 性别、k000002 - 年龄、k000004 - 在网时长、k003886 - 稳定度得分、k000046 - 省分、k003790 - 用户等级（预评级）
basicinfo = ['k000001', 'k000002', 'k000004', 'k003886', 'k000046', 'k003790']

# 2、业务信息指标
# k000060 - 合约类型、k000061 - 合约时长、k000065 - 套餐类型、# k002036 - 是否达到第一次限速未到第二次限速、k002192 - 是否融合、k003798 - 使用网络类型判断、k003817 - 是否智慧沃家
yewuinfo = ['k000060', 'k000061', 'k000065', 'k002036', 'k002192', 'k003798', 'k003817']

# 3、终端信息指标
# k002047 - 上个终端开始时间、k002048 - 上个终端使用时长、k002049 - 上个终端品牌、k002051 - 上个终端价格
# k002053 - 历史换机次数、k002054 - 平均换机间隔、k002205 - 终端是否支持联通Volte、k003637 - 当前使用终端时长
terminalinfo = ['k002047', 'k002048', 'k002049', 'k002051', 'k002053', 'k002054', 'k002205', 'k003637']

#  4、业务使用行为指标
# k000076 - 本地语音主叫通话时长、k002247 - 近三月漫游天数、k002262 - 是否5G用户、k002263 - 套餐金额、k003611 - 总出账金额_月、
# k003725 - 用户近三月月均出账费用、k003726 - 用户近六月月均出账费用、k003727 - 用户近三月月均上网流量、k003728 - 用户近六月月均上网流量
# k003729 - 用户近三月月均通话时长、k003730 - 用户近六月月均通话时长、k003731 - 当月累计-使用总流量、k003747 - 上行点对点短信条数(条)、
# k003856 - 前3个月平均主叫通话时长、k003857 - 前3个月平均漫游通话次数、k003858 - 前6个月平均漫游通话次数
# k003859 - 前6个月平均主叫通话时长
consumeinfo = ['k000076', 'k002247', 'k002262', 'k002263', 'k003611', 'k003725', 'k003726', 'k003727', 'k003728',
               'k003729', 'k003730', 'k003731', 'k003747', 'k003751', 'k003752',
               'k003856', 'k003857', 'k003858', 'k003859']


class DataProcess:
    def __init__(self, data):
        self.data = data

    # 计算两个日期相差多少天
    def Caltime(self, date1, date2):
        date1 = time.strptime(date1, "%Y-%m-%d")
        date2 = time.strptime(date2, "%Y-%m-%d")
        date1 = datetime.datetime(date1[0], date1[1], date1[2])
        date2 = datetime.datetime(date2[0], date2[1], date2[2])
        return (date2 - date1).days

    # 剔除异常值-应用均值填充（应用箱线图的方法进行）
    def Remove_outlier(self, data, col, scale=3):

        tmp = data[col]
        q1 = tmp.quantile(0.25)
        q3 = tmp.quantile(0.75)
        iqr = q3 - q1

        tmp[tmp < (q1 - scale * iqr)] = max(0, q1 - scale * iqr)
        tmp[tmp > (q3 + scale * iqr)] = q3 + scale * iqr
        return tmp

    def Preprocess(self):
        data = self.data

        # 预处理基础信息属性
        def Preprocess_basicinfo(data):
            # 性别应用-1填充缺失值
            data['k000001'] = data['k000001'].fillna(-1)
            # 对于年龄小于0 或者大于80的异常数据应用全网平均值39替换
            data['k000002'] = data['k000002'].apply(lambda x: 39 if x <= 0 or x > 80 else x)
            # 将用户等级中的** 填充为0
            data['k003790'][data['k003790'] == '**'] = 0
            data['k003790'] = data['k003790'].fillna(0)
            data['k003790'] = data['k003790'].astype(np.int32)

            # 稳定度应用平均值进行填充
            data['k003886'] = data['k003886'].fillna(data['k003886'].mean())

            # 对省份id字段进行独热编码
            tmp1 = pd.get_dummies(data['k000046'])
            basicinfo.remove('k000046')
            basicinfo.extend(tmp1.columns.tolist())
            data = pd.concat([data, tmp1], axis=1)
            return data

        def Preprocess_yewuinfo(data):
            # 对k000060、k000061和k002036 中的缺失值进行填充
            data['k000060'] = data['k000060'].fillna(-1)
            data['k000061'] = data['k000061'].fillna(-1)
            data['k002036'] = data['k002036'].fillna(-1)

            # k002036 - 是否达到第一次限速未到第二次限速
            data['k002036'] = data['k002036'].apply(lambda x: 1 if x == '是' else x)
            data['k002036'] = data['k002036'].apply(lambda x: 0 if x == '否' else x)
            return data

        def Preprocess_terminalinfo(data):
            data['k003626'] = data['k003626'].apply(
                lambda x: self.Caltime(str(x), datetime.datetime.now().strftime('%Y-%m-%d')) / 30)
            #  k002049 - 上个终端品牌 保留top20个终端品牌
            term_factory = ['欧珀', '维沃', '华为', '苹果', '小米', '三星', '荣耀', '魅族', '金立', '诺基亚', '乐视', '联想', '步步高', '酷派', '中兴',
                            '锤子', '宏达', '努比亚', '360']
            # 填充空值
            data['k002049'] = data['k002049'].fillna(-1)
            data['k002049'] = data['k002049'].apply(lambda x: x if x in term_factory else 0)
            data['k002049'] = data['k002049'].apply(lambda x: str(x))
            tmp1 = pd.get_dummies(data['k002049'])
            tmp1.rename(columns={'欧珀': 'oppo', '维沃': 'vivo', '华为': 'huawei', '苹果': 'apple', '小米': 'mi', '三星': 'sanxing',
                                 '荣耀': 'honor',
                                 '魅族': 'meizu', '金立': 'jinli', '诺基亚': 'nokia', '乐视': 'leshi', '联想': 'lenovo',
                                 '步步高': 'bubugao',
                                 '酷派': 'kupai', '中兴': 'zet', '锤子': 'smartisan', '宏达': 'hongda', '努比亚': 'nubia',
                                 '360': 'sll'}, inplace=True)
            terminalinfo.remove('k002049')
            terminalinfo.extend(tmp1.columns.tolist())
            data = pd.concat([data, tmp1], axis=1)

            return data

        # 预处理  4、业务使用行为指标
        # k000076 - 本地语音主叫通话时长、k002247 - 近三月漫游天数、k002262 - 是否5G用户、k002263 - 套餐金额、k003611 - 总出账金额_月、
        # k003725 - 用户近三月月均出账费用、k003726 - 用户近六月月均出账费用、k003727 - 用户近三月月均上网流量、k003728 - 用户近六月月均上网流量
        # k003729 - 用户近三月月均通话时长、k003730 - 用户近六月月均通话时长、k003731 - 当月累计-使用总流量、k003747 - 上行点对点短信条数(条)、
        # k003751 - 通话费用(元)、k003752 - 通话时长(分钟)、k003856 - 前3个月平均主叫通话时长、k003857 - 前3个月平均漫游通话次数、k003858 - 前6个月平均漫游通话次数
        # k003859 - 前6个月平均主叫通话时长
        def Preprocess_consumeinfo(data):

            # 异常值处理 k002247 - 近三月漫游天数、k002262
            for i in consumeinfo:
                if i not in ['k002247', 'k002262']:
                    data[i] = self.Remove_outlier(data, i)

            # 用户近三月月均上网流量/用户近六月月均上网流量
            data['k003727'] = data.apply(
                lambda x: x['k003727'] / x['k003728'] if pd.isna(x['k003728']) and pd.isna(x['k003727']) else -1,
                axis=1)
            consumeinfo.remove('k003728')

            # 用户近三月月均出账费用/用户近六月月均出账费用
            data['k003725'] = data.apply(
                lambda x: x['k003725'] / x['k003726'] if pd.isna(x['k003725']) and pd.isna(x['k003726']) else -1,
                axis=1)
            consumeinfo.remove('k003726')

            # 用户近三月月均通话时长/用户近六月月均通话时长
            data['k003729'] = data.apply(
                lambda x: x['k003729'] / x['k003730'] if pd.isna(x['k003729']) and pd.isna(x['k003730']) else -1,
                axis=1)
            consumeinfo.remove('k003730')

            # 前3个月平均主叫通话时长/前6个月平均主叫通话时长
            data['k003856'] = data.apply(
                lambda x: x['k003856'] / x['k003859'] if pd.isna(x['k003856']) and pd.isna(x['k003859']) else -1,
                axis=1)
            consumeinfo.remove('k003859')

            # k003751 - 通话费用(元)/k003752 - 通话时长(分钟)
            data['k003751'] = data.apply(
                lambda x: x['k003751'] / x['k003752'] if pd.isna(x['k003751']) and pd.isna(x['k003752']) else -1,
                axis=1)
            consumeinfo.remove('k003752')
            return data

        data = Preprocess_basicinfo(data)
        print('basicinfo预处理完毕')
        data = Preprocess_yewuinfo(data)
        print('yewuinfo预处理完毕')
        data = Preprocess_terminalinfo(data)
        print('terminalinfo预处理完毕')
        data = Preprocess_consumeinfo(data)
        print('consumeinfo预处理完毕')

        # 对于缺失数据应用-1进行填充
        data = data.fillna(-1)

        # 进行数据标准化处理
        # standscale = StandardScaler()
        # standscale.fit(data)
        # data = standscale.transform(data)

        return data


process = DataProcess(data)
data = process.Preprocess()

from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

x_col = basicinfo + yewuinfo + terminalinfo + consumeinfo
x_col.remove('k002051')
data = data.fillna(-1)
X_train, X_test, y_train, y_test = train_test_split(data[x_col], data['is_5g'], test_size=0.2, random_state=0)

# X_train_apart,X_dev_apart,y_train_apart,y_dev_apart = train_test_split(X_train,y_train,test_size = 0.33,random_state = 2020)

params = {
    'max_depth': 9,
    'n_estimator': 1000,
    'learning_rate': 0.2,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'silent': 0,
    'objective': 'binary:logistic',
    'verbose_eval': 100,
    'n_jobs': -1,
    'colsample_bytree': 0.8,
    'reg_lambda': 0.6
}

folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

for train_index, valid_index in folds.split(X_train, y_train):
    xtrain, ytrain = X_train.iloc[train_index], y_train.iloc[train_index]
    xval, yval = X_train.iloc[valid_index], y_train.iloc[valid_index]

    xgbtrain = xgb.DMatrix(xtrain, label=ytrain)
    xgbval = xgb.DMatrix(xval, label=yval)

    xgb_clf = xgb.XGBClassifier(params=params, )
    xgb_clf.fit(xtrain, ytrain)

    pred = xgb_clf.predict(xval)
    print(accuracy_score(yval, pred))
    print(recall_score(yval, pred))
    print(roc_auc_score(yval, pred))
