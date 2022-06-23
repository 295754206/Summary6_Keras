import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

np.random.seed(7)

df = pd.read_csv("R-2 boston_housing.csv")
print(df.head())
print(df.shape)

dataset = df.values
np.random.shuffle(dataset)

X = dataset[:, 0:13]
Y = dataset[:, 13]
X -= X.mean(axis=0)
X /= X.std(axis=0)
X_train, Y_train = X[:404], Y[:404]
X_test, Y_test = X[404:], Y[404:]


def build_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(X_train.shape[1],), activation="relu"))
    model.add(Dense(16, activation="relu"))  # 增加一层发现效果不错
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return model


# 当使用k-fold交叉验证找出最佳模型后，开始使用全部资料来训练----------------↓

# # 执行 K折交叉验证（K-fold Cross Validation）
# # 若K为4，数据分为4折，第一次用第1折作为验证，第二次用第二折，以此类推
#
# k = 4
# nb_val_samples = len(X_train) // k  # 每一折的样本数，//是向下取整
# nb_epochs = 80
# mse_scores = []
# mae_scores = []
# model = None
#
# # MSE:平均平方误差，MAE:平均绝对误差
#
# for i in range(k):
#     print("Processing Fold: " + str(i))
#     X_val = X_train[i * nb_val_samples:(i + 1) * nb_val_samples]
#     Y_val = Y_train[i * nb_val_samples:(i + 1) * nb_val_samples]
#     X_train_p = np.concatenate(  # 合并函数
#         [X_train[:i * nb_val_samples], X_train[(i + 1) * nb_val_samples:]],
#         axis=0
#     )
#     Y_train_p = np.concatenate(
#         [Y_train[:i * nb_val_samples], Y_train[(i + 1) * nb_val_samples:]],
#         axis=0
#     )
#
#     model = build_model()
#     model.fit(X_train_p, Y_train_p, epochs=nb_epochs, batch_size=16, verbose=0)
#     mse, mae = model.evaluate(X_val, Y_val)
#     mse_scores.append(mse)
#     mae_scores.append(mae)
#
# print("训练过程中的数据：")
# print("MSE_val: ", np.mean(mse_scores))
# print("MAE_val: ", np.mean(mae_scores))
# mse, mae = model.evaluate(X_test, Y_test)
# print("验证的数据：")
# print("MSE_test: ", mse)
# print("MAE_test: ", mae)  # mae为3表示误差为300美金

# ------------------------------------------------------------------↑

model = build_model()
model.fit(X_train, Y_train, epochs=80, batch_size=16, verbose=0)
mse, mae = model.evaluate(X_test, Y_test)
print("MSE_Test: ", mse)
print("MAE_Test: ", mae)