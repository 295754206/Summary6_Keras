import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

np.random.seed(7)

# 步骤一、读取预处理数据

df_train = pd.read_csv("R-4-1 titanic_train.csv")
df_test = pd.read_csv("R-4-2 titanic_test.csv")

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

# 步骤二、定义模型

model = Sequential()
model.add(Dense(11, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(11, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# 步骤三、编译模型

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 步骤四、训练模型

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=10)

# 步骤五、评估模型

loss, accuracy = model.evaluate(X_train, Y_train)
print("训练数据集的准确度：{:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test)
print("测试数据集的准确度：{:.2f}".format(accuracy))

# 步骤六、显示图表来分析训练过程

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

plt.plot(epochs, acc, "b-", label="Training Accuracy")
plt.plot(epochs, val_acc, "r--", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 步骤七、使用全部训练资料来训练模型

file = [df_train, df_test]
file = pd.concat(file)
df = file.values
X = df[:, 0:9]
Y = df[:, 9]

print("\n使用全部资料开始训练...")
model.fit(X, Y, epochs=18, batch_size=10, verbose=0)  # 由上面图表看出18是最佳训练周期

loss, accuracy = model.evaluate(X, Y)
print("训练资料准确度：{:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test)
print("测试资料准确度：{:.2f}".format(accuracy))

# 步骤八、保存神经网络

model.save("Model.h5")
print("神经网络已存储")
