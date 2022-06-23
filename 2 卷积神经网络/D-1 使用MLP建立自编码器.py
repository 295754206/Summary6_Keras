import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

seed = 7
np.random.seed(7)

(X_train, _), (X_test, _) = mnist.load_data()  # 非监督式不需要标签资料，用_来代替

# 步骤一、资料预处理

X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype("float32")
X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype("float32")
X_train = X_train / 255
X_test = X_test / 255

# 步骤二、定义模型

input_img = Input(shape=(784,))
x = Dense(128, activation="relu")(input_img)
encoded = Dense(64, activation="relu")(x)
x = Dense(128, activation="relu")(encoded)
decoded = Dense(784, activation="sigmoid")(x)

autoencoder = Model(input_img, decoded)  # 自编码器模型（AE）
encoder = Model(input_img, encoded)  # 编码器模型

decoder_input = Input(shape=(64,))
decoder_layer = autoencoder.layers[-2](decoder_input)
decoder_layer = autoencoder.layers[-1](decoder_layer)
decoder = Model(decoder_input, decoder_layer)  # 解码器模型

# 步骤三、编译模型

autoencoder.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 步骤四、训练模型

autoencoder.fit(X_train, X_train,  # 训练资料和标签资料都是自己
                validation_data=(X_test, X_test),
                epochs=10,
                batch_size=256,
                shuffle=True,
                verbose=2)

# 步骤五、使用自编码器来编码和解码手写数字图片

encoded_imgs = encoder.predict(X_test)  # 编码图片（也就是压缩图片）
decoder_imgs = decoder.predict(encoded_imgs)

# 绘制图片

n = 10
plt.figure(figsize=(20, 6))  # 指定图片的宽和高

for i in range(n):  # 分三层分别绘制出原图、编码图、解码图
    ax = plt.subplot(3, n, i + 1)  # 行、列、索引: 3行10列中第一个图
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.axis("off")

    ax = plt.subplot(3, n, i + 1 + n)
    ax.imshow(encoded_imgs[i].reshape(8, 8), cmap="gray")
    ax.axis("off")

    ax = plt.subplot(3, n, i + 1 + 2 * n)
    ax.imshow(decoder_imgs[i].reshape(28, 28), cmap="gray")

plt.show()
