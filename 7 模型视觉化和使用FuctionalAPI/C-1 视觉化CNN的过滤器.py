import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)

# 载入图片、将图片转换为4D张量、正规化

(X_train, Y_train), (_, _) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
X_train = X_train / 255  # 因为是固定范围，所以执行正规化，从0-255转为0-1

# 建立模型

model = load_model("R-3 mnist.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 显示神经层数

print("神经层数: ", len(model.layers))
for i in range(len(model.layers)):
    print(i, model.layers[i].name)

# 显示第1个 Conv2D 的 filters 的形状

print(model.layers[0].get_weights()[0].shape)  # (5,5,1,16)表示5x5共16个过滤器

# 绘制第1个 Conv2D 的过滤器

weights = model.layers[0].get_weights()[0]
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(weights[:, :, 0, i], cmap="gray", interpolation="none")
    plt.axis("off")
plt.show()

# 显示第2个 Conv2D 的 filters 的形状

print(model.layers[2].get_weights()[0].shape)  # (5,5,1,16)表示5x5共16个过滤器

# 绘制第2个 Conv2D 的过滤器

weights = model.layers[2].get_weights()[0]
for i in range(32):
    plt.subplot(6, 6, i + 1)
    plt.imshow(weights[:, :, 0, i], cmap="gray", interpolation="none")
    plt.axis("off")
plt.show()
