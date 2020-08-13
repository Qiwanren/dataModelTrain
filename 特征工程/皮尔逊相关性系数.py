from math import sqrt

'''
    用于度量两个变量X与Y之间的相关性（是否线性相关）,其介于 -1 与 1 之间，系数值为1意味着X和Y可以很好的由直线方程来描述，
    所有的数据点都很好的落在一条直线上，且Y随着X的增加而增加，系数为-1则意味着所有的数据点都落在直线上，且Y随着X的增加而减少，
    系数为0则意味着两个变量之间没有线性关系
    
'''

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


x = [0, 1, 0, 3]
y = [0, 1, 1, 1]

print(corrcoef(x, y))  # 0.471404520791