import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

np.random.seed(7)

df_test = pd.read_csv("R-4-2 titanic_test.csv")
dataset_test = df_test.values

X_test = dataset_test[:, 0:9]
Y_test = dataset_test[:, 9]
X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)

model = Sequential()
model = load_model("Model.h5")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("\nTesting ...")
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("测试资料准确度 = {:.2f}".format(accuracy))

# 预测

Y_pre = model.predict_classes(X_test)
print(Y_pre[:, 0])
print(Y_test.astype(int))

# 使用混淆矩阵分析

tb = pd.crosstab(Y_test.astype(int), Y_pre[:, 0], rownames=["label"], colnames=["predict"])
print(tb)
