import numpy as np
import pandas as pd

np.random.seed(7)

df_train = pd.read_csv("R-1 Google_stock_train.csv")
print(df_train.head())

# Adj Close是调整后的收盘价，需要单独取出

X_train_set = df_train.iloc[:, 4:5].values  # date栏不是特征，特征是从Open开始的
X_train_len = len(X_train_set)
print("条数：", X_train_len)


# 制造预测和标签资料

def create_dateset(ds, look_back=1):  # 第一个参数是股价的numpy阵列，第二个参数是回看的天数
    x_data, y_data = [], []
    for i in range(len(ds) - look_back):
        x_data.append(ds[i:(i + look_back), 0])  # 以60天为例，这里产生从i起60天的资料，共158-60=1198笔这样的资料，“,0”将每个[321]改变成了321，即列表的升维
        y_data.append(ds[i + look_back, 0])  # 以60天为例，这里产生61天的资料

    return np.array(x_data), np.array(y_data)  # 注意：股价预测是以 前60天完整资料 来预测 61天的资料，这里是时间序列和前面不一样


look_back = 60
X_train, Y_train = create_dateset(X_train_set, look_back)
print("回看天数：", look_back)
print("X_train Shape: ", X_train.shape)
print("Y_train Shape: ", Y_train.shape)

# 显示前2笔特征资料和第1笔标签资料

print(X_train[0])
print(X_train[1])
print(Y_train[0])






