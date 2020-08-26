from sklearn.preprocessing import LabelEncoder

def fenleiFeatureLabel():
    le = LabelEncoder()
    city_list = ["paris", "paris", "tokyo", "amsterdam","beijing","shanghai"]
    le.fit(city_list)
    print(le.classes_)  # 输出为：['amsterdam' 'paris' 'tokyo']
    city_list_le = le.transform(city_list)  # 进行Encode
    print(city_list_le)  # 输出为：[1 1 2 0]
    city_list_new = le.inverse_transform(city_list_le)  # 进行decode
    print(city_list_new)  # 输出为：['paris' 'paris' 'tokyo' 'amsterdam']


def mothed01():
    str = ''
    strs = str.split('.')
    flag = False
    for s in strs:
        if s.isdigit() == False:
            print(s)
            flag = False
            break
        else:
            print(s)
            flag = True
    return flag

def method02():
    x = 12560
    strs = str(x).split('.')
    return int(strs[0])

def method03():
    from builtins import str
    n = 0.15283333333333332
    n = 0.1
    num = str(n)
    strs = num.split('.')
    print(type(strs))
    len_str = 0
    if len(strs[1]) <=4:
        len_str = len(strs[0]) + len(strs[1]) + 1
    else:
        len_str = len(strs[0]) + 4
    print(num[0:len_str])

import re
def method04():
    str1 = "0.15283333333333332"
    strs = str1.split(".")
    for str in strs:
        print(re.match(r"d+$", str))

def method05():
    prov = "089"
    print(len(prov))
    print(int(prov))

if __name__ == '__main__':
    method05()