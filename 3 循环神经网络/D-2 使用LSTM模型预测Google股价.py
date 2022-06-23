import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

np.random.seed(10)

df_train = pd.read_csv("R-1 Google_stock_train.csv", index_col="Date", parse_dates=True)
df_test = pd.read_csv("R-2 Google_stock_test.csv")
X_train_set = df_train.iloc[:, 4:5].values
X_test_set = df_test.iloc[:, 4:5].values

sc = MinMaxScaler()
X_train_set = sc.fit_transform(X_train_set)  # 使用sc物件执行特征标准化的正规化
X_test_set = sc.fit_transform(X_test_set)


def create_dataset(ds, time_step=1):
    x_data, y_data = [], []
    for i in range(len(ds) - time_step):
        x_data.append(ds[i:(i + time_step), 0])  # 降维
        y_data.append(ds[i + time_step, 0])
    return np.array(x_data), np.array(y_data)


look_back = 60
print("回看天数:", look_back)
X_train, Y_train = create_dataset(X_train_set, look_back)  # 切割训练资料和标签资料
X_test, Y_test = create_dataset(X_test_set, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # 重设为（样本数、时步、特征），最后一个1又升维度
print("X_train.shape: ", X_train.shape)
print("Y_train.shape: ", Y_train.shape)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))

# 建立模型

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

# 编译和训练模型

model.compile(loss="mse", optimizer="adam")
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# 预测股价

X_test_pred = model.predict(X_test)
X_test_pred = sc.inverse_transform(X_test_pred)  # 反转回正常价格
Y_test = sc.inverse_transform(Y_test)
print("X_test_pred:", X_test_pred[0])

# 绘图
print(Y_test)
plt.plot(Y_test, color="red", label="Real Stock Price")
plt.plot(X_test_pred, color="blue", label="Predicted Stock Price")
plt.title("2017 Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Time Price")
plt.legend()
plt.show()
