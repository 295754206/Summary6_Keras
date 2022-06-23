import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")

X_train = X_train / 255
X_test = X_test / 255
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# 定义模型

mnist_input = Input(shape=(28, 28, 1), name="input")
conv1 = Conv2D(16, kernel_size=(5, 5), padding="same", activation="relu", name="conv1")(mnist_input)
pool1 = MaxPooling2D(pool_size=(2, 2), name="pool1")(conv1)
conv2 = Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu", name="conv2")(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2), name="pool2")(conv2)
drop1 = Dropout(0.5, name="drop1")(pool2)
flat = Flatten(name="flat")(drop1)
hidden1 = Dense(128, activation="relu", name="hidden1")(flat)
drop2 = Dropout(0.5, name="drop2")(hidden1)
output = Dense(10, activation="softmax", name="output")(drop2)
model = Model(inputs=mnist_input, outputs=output)

# 输出模型图

plot_model(model, to_file="O-2 model.png", show_shapes=True)

# 编译模型

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练模型

history = model.fit(
    X_train,
    Y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=128,
    verbose=2
)

# 评估模型

loss, accuracy = model.evaluate(X_train, Y_train, verbose=0)
print("训练资料准确度：{:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test)
print("测试资料准确度：{:.2f}".format(accuracy))

# 绘制图表

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
