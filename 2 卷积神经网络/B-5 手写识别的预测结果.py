import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

seed = 7
np.random.seed(seed)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

model = Sequential()
model = load_model("model.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# <A> 使用混淆矩阵分析预测结果

X_test_1 = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")
X_test_1 = X_test_1 / 255
Y_test_bk = Y_test.copy()  # 备份

Y_pred = model.predict_classes(X_test_1)
tb = pd.crosstab(Y_test_bk.astype(int), Y_pred.astype(int), rownames=["label"], colnames=["predict"])
print(tb)

# <B> 绘出0-9数字的预测几率

i = np.random.randint(0, len(X_test))  # 指定一个随机数或者直接命名i=7

digit = X_test[i].reshape(28, 28)  # 用于绘制图片用
plt.figure()
plt.subplot(1, 2, 1)
plt.title("Example of Digit: " + str(Y_test[i]))
plt.imshow(digit, cmap="gray")
plt.axis("off")

X_test_digit = X_test[i].reshape(1, 28, 28, 1).astype("float32")
X_test_digit = X_test_digit / 255
probs = model.predict_proba(X_test_digit, batch_size=1)  # 获取几率组
print(probs)
plt.subplot(1, 2, 2)
plt.title("Probability for each Digit Class")
plt.bar(np.arange(10), probs.reshape(10), align="center")  # 绘制长条图，probs是二维数组，需要转换为10个元素的一维数组
plt.xticks(np.arange(10), np.arange(10).astype(str))  # 第一个参数是x轴的刻度范围，第二个参数是x轴的每个刻度标签的label
plt.show()

# <C> 从错误预测中取样并绘制出预测几率分布图

X_test_3 = X_test.copy()
X_test_3 = X_test_3.reshape(X_test.shape[0], 28, 28, 1).astype("float32")
X_test_3 = X_test_3 / 255

Y_pred = model.predict_classes(X_test_3)  # 获取分类结果
Y_probs = model.predict_proba(X_test_3)  # 获取预测几率分布

df = pd.DataFrame({"label": Y_test, "predict": Y_pred})
df = df[Y_test != Y_pred]  # 获取错误预测的数据组
print(df.head())  # 输出前五个，df.head().to_html("Ch8_4b.html")

i = df.sample(n=1).index.values.astype(int)[0]  # sample函数速记取出一个错误数据组，取其索引
print("Index: ", i)

digit = X_test_3[i].reshape(28, 28)  # 绘制该数字
plt.figure()
plt.subplot(1, 2, 1)
plt.title("Example of Digit:" + str(Y_test[i]))
plt.imshow(digit, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)  # 绘制几率条状图
plt.title("Probabilities for Each Digit Class")
plt.bar(np.arange(10), Y_probs[i].reshape(10), align="center")
plt.xticks(np.arange(10), np.arange(10).astype(str))
plt.show()
