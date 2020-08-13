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

if __name__ == '__main__':
    fenleiFeatureLabel()