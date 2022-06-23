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

# 载入结构与权重、验证（一次性打包版本）

from tensorflow.keras.models import load_model

model = Sequential()
model = load_model("Model.h5")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

loss, accuracy = model.evaluate(X, Y)     # 恢复后是已经训练过的模型，可以直接评估
print("准确度={:.2f}".format(accuracy))

# 载入结构与权重、验证（分开版本）

# from tensorflow.keras.models import model_from_json
#
# model = Sequential()
# with open("Model_Struct.config", 'r') as text_file:
#     json_str = text_file.read()
# model = model_from_json(json_str)
#
# model.load_weights("Model_Weights.weight", by_name=False)
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#
# loss, accuracy = model.evaluate(X, Y)
# print("准确度={:.2f}".format(accuracy))
