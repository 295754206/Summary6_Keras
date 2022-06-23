import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical

seed = 7
np.random.seed(seed)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 建立2个数据集，一个数字小于5，一个数字大于等于5

# 训练资料集

X_train_lt5 = X_train[Y_train < 5]
X_train_gt5 = X_train[Y_train >= 5]

Y_train_lt5 = Y_train[Y_train < 5]
Y_train_gt5 = Y_train[Y_train >= 5] - 5

# 测试资料集

X_test_lt5 = X_test[Y_test < 5]
X_test_gt5 = X_test[Y_test >= 5]

Y_test_lt5 = Y_test[Y_test < 5]
Y_test_gt5 = Y_test[Y_test >= 5] - 5

# 将训练资料集图片转换为 4D 张量

X_train_lt5 = X_train_lt5.reshape((X_train_lt5.shape[0], 28, 28, 1)).astype("float32")
X_train_gt5 = X_train_gt5.reshape((X_train_gt5.shape[0], 28, 28, 1)).astype("float32")
X_test_lt5 = X_test_lt5.reshape((X_test_lt5.shape[0], 28, 28, 1)).astype("float32")
X_test_gt5 = X_test_gt5.reshape((X_test_gt5.shape[0], 28, 28, 1)).astype("float32")

# 固定范围，进行正规化

X_train_lt5 = X_train_lt5 / 255
X_test_lt5 = X_test_lt5 / 255
X_train_gt5 = X_train_gt5 / 255
X_test_gt5 = X_test_gt5 / 255

# One-hot编码

Y_train_lt5 = to_categorical(Y_train_lt5, 5)
Y_test_lt5 = to_categorical(Y_test_lt5, 5)
Y_train_gt5 = to_categorical(Y_train_gt5, 5)
Y_test_gt5 = to_categorical(Y_test_gt5, 5)

# 定义模型

# 卷积基底：2组卷积+池化

model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 分类器

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(5, activation="softmax"))

# 编译模型

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练数字小于5的模型

history = model.fit(
    X_train_lt5,
    Y_train_lt5,
    validation_split=0.2,
    epochs=5,
    batch_size=128,
    verbose=2
)

# 评估数字小于5的模型

loss, accuracy = model.evaluate(X_test_lt5, Y_test_lt5, verbose=0)
print("数字小于5的测试资料集准确度：{:.2f}".format(accuracy))

# 显示各神经层

print(len(model.layers))
for i in range(len(model.layers)):
    print(i, model.layers[i])

# 冻结卷积基底的权重，不再参与训练

for i in range(4):
    model.layers[i].trainable = False

# 注意仍需要再次编译

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练数字大于5的模型

history = model.fit(
    X_train_gt5,
    Y_train_gt5,
    validation_split=0.2,
    epochs=5,
    batch_size=128,
    verbose=2
)

# 评估数字大于5的模型

loss, accuracy = model.evaluate(X_test_gt5, Y_test_gt5, verbose=0)
print("测试资料集的准确度：{:.2f}".format(accuracy))
