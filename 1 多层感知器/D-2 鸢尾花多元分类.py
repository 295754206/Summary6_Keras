import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

np.random.seed(7)

# 步骤一、资料预处理

df = pd.read_csv("R-3 iris_data.csv")
target_mapping = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
}
df["target"] = df["target"].map(target_mapping)

dataset = df.values
np.random.shuffle(dataset)
X = dataset[:, 0:4].astype(float)
Y = to_categorical(dataset[:, 4])
X -= X.mean(axis=0)
X /= X.std(axis=0)

X_train, Y_train = X[:120], Y[:120]
X_test, Y_test = X[:120], Y[:120]

# 步骤二、定义模型

model = Sequential()
model.add(Dense(6, input_shape=(4,), activation="relu"))
model.add(Dense(6, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.summary()

# 步骤三、编译模型

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])  # 定义损失函数、优化器、评估标准

# 步骤四、训练模型

model.fit(X_train, Y_train, epochs=100, batch_size=5)

# 步骤五、评估模型

loss, accuracy = model.evaluate(X_test, Y_test)
print("准确度：{:.2f}".format(accuracy))

# 步骤六、存储模型

model.save("Model.h5")
print("模型已存储")