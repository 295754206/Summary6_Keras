import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

np.random.seed(10)

# 步骤一：资料预处理

df = pd.read_csv("./R-1 diabetes.csv")
dataset = df.values
np.random.shuffle(dataset)

X = dataset[:, 0:8]
Y = dataset[:, 8]

X -= X.mean(axis=0)
X /= X.std(axis=0)

X_train, Y_train = X[:690], Y[:690]
X_test, Y_test = X[690:], Y[690:]

# 步骤二：定义模型

model = Sequential()
model.add(Dense(10, input_shape=(8,), activation="relu"))
model.add(Dense(6, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 步骤三：编译模型

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# 步骤四：训练模型

# model.fit(X_train, Y_train, epochs=150, batch_size=10, verbose=0)

# 在训练的时候就使用验证数据并显示校正过程
history = model.fit(X_train, Y_train,
                    validation_data=(X_test, Y_test),  # validation_split=0.2，自动从训练资料中切割出来20%作为验证，另一个常用数值是0.33
                    epochs=10,
                    batch_size=10)

# 步骤五：评估模型

# loss, accuracy = model.evaluate(X_train, Y_train)
# print("训练资料 的准确度={:.2f}".format(accuracy))
#
# loss, accuracy = model.evaluate(X_test , Y_test)
# print("测试资料 的准确度={:.2f}".format(accuracy))

# 训练准确度0.84，测试准确度0.69，说明有过度拟合的问题

# 绘制出图形

# 损失

train_loss = history.history["loss"]
valid_loss = history.history["val_loss"]

epochs = range(1, len(train_loss) + 1)

plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.plot(epochs, train_loss, "bo", label="Training Loss")
plt.plot(epochs, valid_loss, "r", label="Validattion Loss")
plt.legend()
plt.show()

# 准确度

train_acc = history.history["accuracy"]
valid_acc = history.history["val_accuracy"]

epochs = range(1, len(train_acc) + 1)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.plot(epochs, train_acc, "b-", label="Training Acc")
plt.plot(epochs, valid_acc, "r--", label="Validattion Acc")
plt.legend()
plt.show()

# 由图可以得：10次最佳，不用150次

# 预测

Y_pred = model.predict(X_test, batch_size=10, verbose=0)
print(Y_pred[0])

# 对于分类预测，可以用predict_classes()来直接给出类别
# Y_pred = model.predict_classes(X_test, batch_size=10, verbose=0)
# print(Y_pred[0],Y_pred[1])
