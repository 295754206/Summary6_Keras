import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt

seed = 10
np.random.seed(seed)

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()


# 打乱资料集

def randomize(a, b):
    permutation = list(np.random.permutation(a.shape[0]))
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]

    return shuffled_a, shuffled_b


X_train, Y_train = randomize(X_train, Y_train)
X_test, Y_test = randomize(X_test, Y_test)

# 取出10%训练，10%测试

X_train = X_train[:5000]
Y_train = Y_train[:5000]
X_test = X_test[:1000]
Y_test = Y_test[:1000]

# One-hot编码

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

# 载入 ResNet50 预训练模型

resnet_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(200, 200, 3)  # 这是输入参数
)

# 调整图片尺寸

X_train_new = np.array(
    [
        np.asarray(
            Image.fromarray(
                X_train[i]
            ).resize(
                (200, 200)
            ))
        for i in range(0, len(X_train))
    ]
)

X_test_new = np.array(
    [
        np.asarray(
            Image.fromarray(
                X_test[i]
            ).resize(
                (200, 200)
            ))
        for i in range(0, len(X_test))])

X_train_new = X_train_new.astype("float32")
X_test_new = X_test_new.astype("float32")

# 资料训练前进行预处理

train_input = preprocess_input(X_train_new)
test_input = preprocess_input(X_test_new)

# 使用 ResNet50 预训练模型来输出特征数据

train_features = resnet_model.predict(train_input)
test_features = resnet_model.predict(test_input)

# 定义第二个模型（分类器）

model = Sequential()
model.add(
    GlobalAveragePooling2D(  # 使用GlobalAveragePooling2D来取代Flatten平坦层来减少模型参数量
        input_shape=train_features.shape[1:]
    )
)
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

# 编译模型

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练模型

history = model.fit(
    train_features,
    Y_train,
    validation_data=(test_features, Y_test),
    epochs=14,
    batch_size=32,
    verbose=2
)

# 评估模型

loss, accuracy = model.evaluate(test_features, Y_test, verbose=0)
print("测试资料的准确度：{:.2f}".format(accuracy))

# 绘制训练过程

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
