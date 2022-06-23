import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

seed = 7
np.random.seed(seed)

# 步骤一、资料的载入和预处理

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# X_train[0]是二维数组格式
# MLP是(X_train.shape[0], 28 * 28) 单个元素一维格式输入（第一个是条数，单个数据就是后面的28*28）
# CNN是(X_train.shape[0], 28, 28, 1) 单个元素三维格式（行，列，灰阶数值）格式输入
# LSTM使用原格式输入，即（行，列）

X_train = X_train / 255
X_test = X_test / 255
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# 步骤二、定义模型

model = Sequential()
model.add(LSTM(28, input_shape=(X_train.shape[1:]), activation="relu", return_sequences=True))  # 第三个参数True表示传回全部序列资料
model.add(LSTM(28, activation="relu"))  # 堆叠2层LSTM层
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

# 步骤三、编译和训练模型

# 这里注意：如果标签资料也就是Y_train,Y_test没有执行One-hot编码的话，且标签是整数值（例如0-9），那么损失函数要改成parse_categorical_crossentropy

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=128, verbose=2)

# 步骤四、评估模型

loss, accuracy = model.evaluate(X_test, Y_test)
print("测试资料的准确度：{:.2f}".format(accuracy))

loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

accuracy = history.history["accuracy"]
epochs = range(1, len(accuracy) + 1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, accuracy, "b-", label="Training Acc")
plt.plot(epochs, val_acc, "r--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
