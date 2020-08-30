# 导入pandas包
import pandas as pd

def method1():
    '''
        分箱操作
            1、指定临界值
                1）、根据临界值进行划分，使用 cut() 函数，只能识别范围内的（18-100）| 无法识别 101 和 17 这样的数据
                2）、设置labels属性
                    1、可以设置特定值，也可设置为Flase , 以数字的形式显示
                        labels参数为False时，返回结果中用不同的整数作为箱子的指示符
                        cats2 = pd.cut(ages, bins,labels=False)
                3）、指定分箱区间是左闭右开
                    pd.cut(ages, [18, 26, 36, 61, 100], right=False)
            2、pd.cut(ages, [18, 26, 36, 61, 100], right=False)   # 指定分箱区间是左闭右开
    '''
    ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]  # 待分箱数据
    bins = [18, 25, 35, 60, 100]  # 指定箱子的分界点
    group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
    ## cats1 = pd.cut(ages, bins)  不指定标签值
    cuts3 = pd.cut(ages, bins, labels=group_names)
    print(cuts3)

def one_hot_code():
    pass



if __name__ == '__main__':
    method1()