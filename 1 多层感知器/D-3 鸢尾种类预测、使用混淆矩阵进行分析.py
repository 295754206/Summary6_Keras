import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

np.random.seed(7)

# 步骤一、资料预处理（载入模型时仍需要先处理数据）

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
X_test, Y_test = X[120:], Y[120:]

# 步骤二、直接载入模型

model = Sequential()
model = load_model("Model.h5")

# 步骤三、编译模型

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 步骤四、训练模型（之前已经训练过，所以载入后不再需要重新训练）

# 步骤五、评估模型

loss, accuracy = model.evaluate(X_test, Y_test)
print("准确度：{:.2f}".format(accuracy))

# 步骤六、预测鸢尾花的种类

Y_pred = np.argmax(model.predict(X_test), axis=-1)
print(Y_pred)
Y_target = dataset[:, 4][120:].astype(int)  # 所有行的第4列，前120条记录
print(Y_target)

# 步骤七、使用混淆矩阵Confusion Matrix进行分析（数据量较大时常用）

tb = pd.crosstab(Y_target, Y_pred, rownames=["label"], colnames=["predict"])
print(tb)
