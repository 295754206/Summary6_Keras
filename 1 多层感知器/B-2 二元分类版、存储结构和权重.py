import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical  # 需要添加的地方

np.random.seed(10)

# 步骤一：资料预处理

df = pd.read_csv("./R-1 diabetes.csv")
dataset = df.values
np.random.shuffle(dataset)

X = dataset[:, 0:8]
Y = dataset[:, 8]
Y = to_categorical(Y)  # 需要添加的地方，Y需要One-hot编码
X -= X.mean(axis=0)
X /= X.std(axis=0)

# 步骤二：定义模型

model = Sequential()
model.add(Dense(10,
                input_shape=(8,),
                kernel_initializer="random_uniform",
                bias_initializer="ones",
                activation="relu"
                ))
model.add(Dense(8,
                kernel_initializer="random_uniform",
                bias_initializer="ones",
                activation="relu"))
model.add(Dense(2,
                kernel_initializer="random_uniform",
                bias_initializer="ones",
                activation="softmax"))  # 2个神经元，更改为softmax
model.summary()

# 步骤三：编译模型

model.compile(loss="binary_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# 步骤四：训练模型

model.fit(X, Y, epochs=150, batch_size=10, verbose=0)

# 步骤五：评估模型

loss, accuracy = model.evaluate(X, Y)
print("准确度={:.2f}".format(accuracy))

# 步骤六：存储权重和结构（同时存储结构和权重版本）

model.save("Model.h5") # HDF5格式

# 步骤六：存储权重和结构（分开存储结构和权重版本）

# json_str = model.to_json()  # 存储结构
# with open("Model_Struct.config", "w") as text_file:
#     text_file.write(json_str)
#
# model.save_weights("Model_Weights.weight")  # 存储权重
