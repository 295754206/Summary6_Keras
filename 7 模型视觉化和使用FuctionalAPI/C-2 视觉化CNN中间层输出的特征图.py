import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D

seed = 7
np.random.seed(seed)

(X_train, Y_train), (_, _) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
X_train = X_train / 255

model = Sequential()
model = load_model("R-3 mnist.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 建立新的 Conv2D 测试模型

model_test = Sequential()
model_test.add(Conv2D(
    16,
    kernel_size=(5, 5),
    padding="same",
    input_shape=(28, 28, 1),
    activation="relu"
))

# 用新的测试模型载入原模型的权重

for i in range(len(model_test.layers)):
    model_test.layers[i].set_weights(model.layers[i].get_weights())

# Functional API 的做法：使用model建立Conv2D层
# 这里使用Functional API重组CNN模型现有的Conv2D层，不需要再指定权重

# from tensorflow.keras.models import Model
# layer_name = "conv2d_1"
# model_test = Model(
#     inputs=model.input,
#     outputs=model.get_layer(layer_name).output
# )

# 用此模型预测训练资料集的第一张图片的特征输出图

output = model_test.predict(X_train[0].reshape(1, 28, 28, 1))  # 与此同时转换为(1,28,28,1)向量

# 绘出第1个 Conv2D 层的特征输出图

plt.figure(figsize=(10, 8))
for i in range(0, 16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(output[0, :, :, i], cmap="gray")
    plt.axis("off")
plt.show()

# 绘制第1层池化层输出的特征图

model_test_2 = Sequential()
model_test_2.add(Conv2D(
    16,
    kernel_size=(5, 5),
    padding="same",
    input_shape=(28, 28, 1),
    activation="relu"
))
model_test_2.add(MaxPooling2D(pool_size=(2, 2)))

for i in range(len(model_test_2.layers)):
    model_test_2.layers[i].set_weights(model.layers[i].get_weights())

# Fuctional API做法

# from tensorflow.keras.models import Model
# layer_name = "max_pooling2d_1"
# model_test_2 = Model(inputs=model.input,
#                    outputs=model.get_layer(layer_name).output)

output = model_test_2.predict(X_train[0].reshape(1, 28, 28, 1))

plt.figure(figsize=(10, 8))
for i in range(0, 16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(output[0, :, :, i], cmap="gray")
    plt.axis("off")
plt.show()
