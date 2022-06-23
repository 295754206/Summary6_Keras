import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

seed = 7
np.random.seed(seed)

(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")
X_train = X_train / 255
X_test = X_test / 255

# 给图片制造杂讯

nf = 0.5

size_train = X_train.shape
X_train_noisy = X_train + nf * np.random.normal(loc=0.0, scale=1.0, size=size_train)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)

size_train = X_test.shape
X_test_noisy = X_test + nf * np.random.normal(loc=0.0, scale=1.0, size=size_train)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

# 定义模型

input_img = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3), activation="relu", padding="same")(input_img)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
encoded = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation="relu")(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()

encoder = Model(input_img, encoded)
encoder.summary()

decoder_input = Input(shape=(4, 4, 8))
decoder_layer = autoencoder.layers[-7](decoder_input)
decoder_layer = autoencoder.layers[-6](decoder_layer)
decoder_layer = autoencoder.layers[-5](decoder_layer)
decoder_layer = autoencoder.layers[-4](decoder_layer)
decoder_layer = autoencoder.layers[-3](decoder_layer)
decoder_layer = autoencoder.layers[-2](decoder_layer)
decoder_layer = autoencoder.layers[-1](decoder_layer)
decoder = Model(decoder_input, decoder_layer)
decoder.summary()

# 编译模型

autoencoder.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练模型

autoencoder.fit(X_train_noisy,
                X_train,
                validation_data=(X_test_noisy, X_test),
                epochs=10,
                batch_size=128,
                shuffle=True,
                verbose=2)

# 压缩图片

encoded_imgs = encoder.predict(X_test_noisy)

# 解压缩图片

decoded_imgs = decoder.predict(encoded_imgs)

# 绘图（杂讯图片、压缩图片、还原图片）

n = 10
plt.figure(figsize=(20, 8))

for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    ax.imshow(X_test_noisy[i].reshape(28, 28), cmap="gray")
    ax.axis("off")

    ax = plt.subplot(3, n, i + 1 + n)
    ax.imshow(encoded_imgs[i].reshape(4, 4 * 8).T, cmap="gray")
    ax.axis("off")

    ax = plt.subplot(3, n, i + 1 + 2 * n)
    ax.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    ax.axis("off")

plt.show()
