import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# 步骤一、数据预处理

seed = 7
np.random.seed(seed)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")

X_train = X_train / 255
X_test = X_test / 255
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# 步骤二、定义模型

model = Sequential()
model.add(Conv2D(16,  # 第一个卷积层16核
                 kernel_size=(5, 5),  # 过滤器大小
                 padding="same",  # 补零方式，预设valid不补零，same是补零成相同尺寸
                 input_shape=(28, 28, 1),
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))  # 沿水平、垂直方向缩小比例额，（2，2）是各缩小一半
model.add(Conv2D(32,
                 kernel_size=(5, 5),
                 padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.summary()
# 卷积层1：1*(5*5)*16+16=416，1是输入：黑白图
# 卷积层2：16*(5*5)*32+32=12832，16是上一层输入：特征图数
# Dense1：1568*128+128=200832
# Dense2：128*10+10=1290

# 步骤三、编译模型

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 步骤四、训练模型

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=128, verbose=2)

# 步骤五、评估模型

loss, accuracy = model.evaluate(X_train, Y_train)
print("训练资料的准确度：{:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test)
print("测试资料的准确度：{:.2f}".format(accuracy))
model.save("model.h5")

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
