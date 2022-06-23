import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)

df_train = pd.read_csv("R-1 titanic_train.csv")
df_test = pd.read_csv("R-2 titanic_test.csv")
dataset_train = df_train.values
dataset_test = df_test.values

X_train = dataset_train[:, 0:9]
Y_train = dataset_train[:, 9]
X_test = dataset_test[:, 0:9]
Y_test = dataset_test[:, 9]

X_train -= X_train.mean(axis=0)
X_train /= X_train.std(axis=0)
X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)

# 定义模型

# 前两个Dense层都没有activation参数
# use_bias为False不使用偏向量，因为执行批次正规化偏向量没有用，只会增加计算量

model = Sequential()
model.add(Dense(11, input_dim=X_train.shape[1], use_bias=False))
model.add(BatchNormalization())  # 增加BN层
model.add(Activation("relu"))
model.add(Dense(11, use_bias=False))
model.add(BatchNormalization())  # 增加BN层
model.add(Activation("relu"))
model.add(Dense(1, activation="sigmoid"))

# 编译模型

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练模型

history = model.fit(
    X_train,
    Y_train,
    verbose=2,
    validation_data=(X_test, Y_test),
    epochs=34,
    batch_size=10
)

# 评估模型

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("测试资料的准确度：{:.2f}".format(accuracy))

# 绘制过程图

loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "b-", label="Training Loss")
plt.plot(epochs, val_loss, "r--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

acc = history.history["accuracy"]
epochs = range(1, len(acc) + 1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "b-", label="Training Acc")
plt.plot(epochs, val_acc, "r--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
