import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

np.random.seed(10)

df = pd.read_csv("R-3 diabetes.csv")
dataset = df.values
np.random.shuffle(dataset)

X = dataset[:, 0:8]
Y = dataset[:, 8]

X -= X.mean(axis=0)
X /= X.std(axis=0)

Y = to_categorical(Y)

X_train, Y_train = X[:690], Y[:690]
X_test, Y_test = X[690:], Y[690:]

# 定义模型

model = Sequential()
model.add(Dense(8, input_shape=(8,), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(2, activation="softmax"))
model.load_weights("O-1 best_model.h5")  # 载入最佳权重

# 编译模型

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 评估模型

loss, accuracy = model.evaluate(X_train, Y_train, verbose=0)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("測試資料集的準確度 = {:.2f}".format(accuracy))
