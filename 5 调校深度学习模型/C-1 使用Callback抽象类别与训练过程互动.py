import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback

np.random.seed(10)

df = pd.read_csv("R-3 diabetes.csv")
dataset = df.values
np.random.shuffle(dataset)

X = dataset[:, 0:8]
Y = dataset[:, 8]

# 定义模型

model = Sequential()
model.add(Dense(10, input_shape=(8,), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 编译模型

model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])


# 使用Callback抽象类别存储训练过程中的准确度和损失值

class fitHistory(Callback):
    def on_train_begin(self, logs={}):  # 开始训练前建立准确度和损失清单
        self.accuracy = []
        self.losses = []

    def on_batch_end(self, batch, logs={}):  # 每轮次后增加至清单
        self.accuracy.append(logs.get("accuracy"))
        self.losses.append(logs.get("loss"))


# 建立Callback物件history

history = fitHistory()
model.fit(
    X,
    Y,
    batch_size=64,
    epochs=5,
    verbose=0,
    callbacks=[history]  # 使用callback物件
)

# 显示Callback记录

print("Accuracy记录数：", len(history.accuracy))
print(history.accuracy)
print("Loss记录数", len(history.losses))
print(history.losses)  # 768/64x5 = 60

# 评估模型

loss, accuracy = model.evaluate(X, Y, verbose=0)
print("准确度：{:.2f}".format(accuracy))
