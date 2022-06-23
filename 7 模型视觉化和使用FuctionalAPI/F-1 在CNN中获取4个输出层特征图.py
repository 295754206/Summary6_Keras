import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

seed = 7
np.random.seed(seed)

(X_train, Y_train), (_, _) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
X_train = X_train / 255

model = Sequential()
model = load_model("R-3 mnist.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 使用Model建立前4层的Conv2D和MaxPooling层（此模型是1个输入，4个输出）

layer_outputs = [layer.output for layer in model.layers[:4]]
model_test = Model(inputs=model.input, outputs=layer_outputs)

# 预测输出

outputs = model_test.predict(X_train[0].reshape(1, 28, 28, 1))

# 取得第一个 Conv2D 的输出并绘制

output = outputs[0]

plt.figure(figsize=(10, 8))
for i in range(0, 16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(output[0, :, :, i], cmap="gray")
    plt.axis("off")

# 取得第一个 MaxPooling 的输出并绘制

output = outputs[1]

plt.figure(figsize=(10, 8))
for i in range(0, 16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(output[0, :, :, i], cmap="gray")
    plt.axis("off")

# 取得第二个 Conv2D 的输出并绘制

output = outputs[2]

plt.figure(figsize=(10, 8))
for i in range(0, 32):
    plt.subplot(6, 6, i + 1)
    plt.imshow(output[0, :, :, i], cmap="gray")
    plt.axis("off")

# 取得第二个 MaxPooling 的输出并绘制

output = outputs[3]

plt.figure(figsize=(10, 8))
for i in range(0, 32):
    plt.subplot(6, 6, i + 1)
    plt.imshow(output[0, :, :, i], cmap="gray")
    plt.axis("off")

plt.show()
