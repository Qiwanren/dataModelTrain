import numpy as np
import pandas as pd
'''
参数依次为：最小值，最大值，分箱长度
生成左闭又开区间
'''
def getBins(minValue,maxValue,len):
    bins = []
    b=0
    a = np.arange(minValue, maxValue, len)
    for a1 in a:
        bin = []
        bin.append(b)
        bin.append(a1)
        b = a1
        bins.append(bin)
    return bins
'''
    左开右闭区间切分
'''
def cutFeautreData(bins,x):
    ret_str = ''
    for bin in bins:
        if x>=bin[0] and bin[1]>x:
            ret_str = '[' + str(bin[0]) + ',' + str(bin[1]) + ']'
            break
        else:
            ret_str = ''
    if ret_str != '':
        return ret_str
    else:
        return '[,]'

def one_hot(dataFrame,feature):
    list(dataFrame)
    dummies = pd.get_dummies(dataFrame[feature], prefix='gender')
    print(dummies)
    df = pd.concat([dataFrame, dummies], axis=1)
    print(df)