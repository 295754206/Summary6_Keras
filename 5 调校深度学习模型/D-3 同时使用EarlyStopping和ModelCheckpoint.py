import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

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

# 编译模型

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 建立 EarlyStopping 物件

es = EarlyStopping(
    monitor="val_loss",
    mode="min",
    verbose=1,
    patience=5
)

# 建立 ModelCheckpoint 物件

filename = "O-2 weights-{epoch:02d}-{val_accuracy:.2f}.h5"  # 会保留所有最佳记录
mc = ModelCheckpoint(
    filename,
    monitor="val_accuracy",
    mode="max",
    verbose=1,
    save_best_only=True
)

# 训练模型

history = model.fit(
    X_train,
    Y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=10,
    verbose=0,
    callbacks=[es, mc]
)

# 评估模型

loss, accuracy = model.evaluate(X_train, Y_train, verbose=0)
print("训练资料的准确度：{:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("测试资料的准确度：{:.2f}".format(accuracy))

# 显示训练过程

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
