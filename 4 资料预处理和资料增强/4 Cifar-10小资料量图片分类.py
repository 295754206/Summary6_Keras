import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf  # 需要加上才能正常执行
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

tf.disable_eager_execution()  # 需要加上才能正常执行

seed = 10
np.random.seed(seed)

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()


# 打乱资料

def randomize(a, b):
    permutation = list(np.random.permutation(a.shape[0]))  # 产生随机索引清单
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b


X_train, Y_train = randomize(X_train, Y_train)
X_test = X_test.astype("float32") / 255
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# 取前20%资料

X_train_part = X_train[:10000]
Y_train_part = Y_train[:10000]
print(X_train_part.shape, Y_train_part.shape)

# 显示每种类型有几笔资料（这里由于已经One-hot化是列表组，90000个0，10000个1，如果不执行One-hot可以正常组合）

# unique, counts = np.unique(Y_train_part, return_counts=True)
# print(unique)
# print(counts)
# print(dict(zip(unique, counts)))

# 进行图片增强

train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # 正规化
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow(  # 产生图片流
    X_train_part,
    Y_train_part,
    batch_size=16
)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding="same", input_shape=X_train.shape[1:], activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=10000,
    epochs=14, verbose=2,
    validation_data=(X_test, Y_test))

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("测试资料的准确度： {:.2f}".format(accuracy))

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

accuracy = history.history["accuracy"]
epochs = range(1, len(accuracy) + 1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, accuracy, "bo-", label="Training Acc")
plt.plot(epochs, val_acc, "ro--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
