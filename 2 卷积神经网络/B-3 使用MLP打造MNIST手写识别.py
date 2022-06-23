import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# 步骤一、资料预处理

seed = 7
np.random.seed(seed)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype("float32")  # 表示重塑成多少行多少列，若有-1表示此参数尚不知道写多少
X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype("float32")
print("X_train Shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)

X_train = X_train / 255  # 灰阶图片是0-255，实施正规化
X_test = X_test / 255
print(X_train[0][150:175])  # 显示第一张图片150-174之间的范围

Y_train = to_categorical(Y_train)  # 手写图片的数字识别是多元分类，执行One-hot编码
Y_test = to_categorical(Y_test)
print("Y_train Shape: ", Y_train.shape)
print(Y_train[0])  # 第一张图片是5

# 步骤二、定义模型

model = Sequential()
model.add(Dense(256, input_dim=784, activation="relu"))  # dim为28*28时产生了过度拟合，所提增加为784形成更宽的MLP，但仍未解决过度拟合问题
model.add(Dropout(0.5))  # 增加Dropout层50%随机归零可以在MLP中解决过度拟合问题
model.add(Dense(256, activation="relu"))  # 新增一层隐藏层，也不能解决过度拟合问题
model.add(Dense(10, activation="softmax"))
model.summary()

# 步骤三、编译模型

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 步骤四、训练模型

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=128, verbose=2)  # 分割验证资料20%

# 步骤五、评估模型

loss, accuracy = model.evaluate(X_train, Y_train)
print("训练资料准确度：{:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test)
print("测试资料准确度：{:.2f}".format(accuracy))

# 绘图

loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

acc = history.history["accuracy"]
epochs = range(1, len(acc) + 1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo-", label="Training Acc")
plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
