import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

seed = 10
np.random.seed(seed)

# 步骤一、资料预处理

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# 步骤二、定义模型

model = Sequential()
model.add(Conv2D(32,  # 输入图片是32X32X3
                 kernel_size=(3, 3),
                 padding="same",
                 input_shape=X_train.shape[1:],
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,
                 kernel_size=(3, 3),
                 padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.summary()
# 第一层：输入色彩数即通道数3，过滤器窗口大小3X3，32核：3x(3x3)x32+32=896
# 第二层：前一层通道数32，过滤器窗口大小3X3，64核：32x(3x3)x64+64=18496
# 第三层：平坦层输入：8x8x64=4096 4096*512+512=2097664
# 第四层：输出层Dense 512*10+10=5130

# 步骤三、编译模型

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 步骤四、训练模型

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=9, batch_size=128, verbose=2)

# 步骤五、评估与训练模型

loss, accuracy = model.evaluate(X_train, Y_train)
print("训练资料集的准确度：{:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test)
print("测试资料集的准确度：{:.2f}".format(accuracy))

# 绘图

loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo-", label="Training Loss")
plt.plot(epochs, val_loss, "ro--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

acc = history.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo-", label="Training Acc")
plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

model.save("model.h5")