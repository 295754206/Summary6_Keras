import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

seed = 10
np.random.seed(seed)

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_test = X_test.astype("float32") / 255
Y_test_bk = Y_test.copy()
Y_test = to_categorical(Y_test)

model = Sequential()
model = load_model("model.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# <A> 使用混淆矩阵分析预测结果

Y_pred = model.predict_classes(X_test)
tb = pd.crosstab(Y_test_bk.astype(int).flatten(),
                 Y_pred.astype(int),
                 rownames=["label"],
                 colnames=["predict"])
print(tb)

# <B> 绘出图片0-9分类的预测几率

i = np.random.randint(0, len(X_test))
img = X_test[i]
X_test_img = img.reshape(1, 32, 32, 3).astype("float32")
X_test_img = X_test_img / 255

plt.figure()
plt.subplot(1, 2, 1)
plt.title("Example of Image:" + str(Y_test[i]))
plt.imshow(img, cmap="binary")
plt.axis("off")

print("Predicting ...")
probs = model.predict_proba(X_test_img, batch_size=1)
plt.subplot(1, 2, 2)
plt.title("Probabilities for Each Image Class")
plt.bar(np.arange(10), probs.reshape(10), align="center")
plt.xticks(np.arange(10), np.arange(10).astype(str))
plt.show()

# <C> 筛选分类错误和绘出错误分类的预测几率

Y_pred = model.predict_classes(X_test)
Y_probs = model.predict_proba(X_test)
Y_test_bk = Y_test_bk.flatten()
df = pd.DataFrame({"label": Y_test_bk, "predict": Y_pred})
df = df[Y_test_bk != Y_pred]
print(df.head())

i = df.sample(n=1).index.values.astype(int)[0]  # 随机取一个错误预测
print("Index: ", i)
img = X_test[i]

plt.figure()
plt.subplot(1, 2, 1)
plt.title("Example of Image:" + str(Y_test[i]))
plt.imshow(img, cmap="binary")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Probabilities for Each Image Class")
plt.bar(np.arange(10), Y_probs[i].reshape(10), align="center")
plt.xticks(np.arange(10), np.arange(10).astype(str))
plt.show()
