from sklearn.preprocessing import StandardScaler  # 导入数据标准化方法

def data_preprocess(df):
    scaler = StandardScaler()
    sca_x = scaler.fit_transform(df.iloc[:, :-1])
    return sca_x



import numpy as np
from collections import deque  # 相当于一个列表，可以在两头增删元素


def  generate_features(df,sca_x,max_len=20,pre_days = 10):
    deq = deque(maxlen=max_len)
    x = []
    for i in sca_x:
        deq.append(list(i))
        if len(deq) == max_len:
            x.append(list(deq))
    x = x[:-pre_days]
    y = df['label'].values[max_len-1: -pre_days]
    print(len(y))  # 序列x和标签y的长度应该一样
    x, y = np.array(x), np.array(y)
    return x,y
