import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

np.random.seed(10)  # 指定乱数

# 步骤一：资料预处理

df = pd.read_csv("./R-1 diabetes.csv")
print(df.head())  # 使用head()函数显示 前5笔记录
print(df.shape)  # 使用shape函数显示有 768行，9列

dataset = df.values  # df是（文件名，数据），现在直接取后面数据
np.random.shuffle(dataset)  # 使用函数打乱纪录的先后次序

X = dataset[:, 0:8]  # 前面是行，后面是列[0,8)
Y = dataset[:, 8]  # 切割资料变为 特征数据 和 标签数据

X -= X.mean(axis=0)  # 优化：特征标准化
X /= X.std(axis=0)

# 步骤二：定义模型
# 输入层：8元（8个特征），隐藏层1：10元，隐藏层2：8元，输出层：1元

model = Sequential()  # 建立Sequential物件
model.add(Dense(10, input_shape=(8,), activation="relu"))  # 增加隐藏层1并输入元数和输入元数，指定激活函数
model.add(Dense(6, activation="relu"))  # 增加隐藏层2，对于样本数不多的数据神经元从8改为6能减少神经网络尺寸提高效能
model.add(Dense(1, activation="sigmoid"))  # 增加输出层

# 指定权重和偏向量的写法，不写的话：权重默认是glorot_uniform，偏向量是zeros

# model.add(Dense(10,
#                 input_shape=(8,),
#                 kernel_initializer="random_uniform",
#                 bias_initializer="ones",
#                 activation="relu"))
# model.add(Dense(8,
#                 kernel_initializer="random_uniform",
#                 bias_initializer="ones",
#                 activation="relu"))
# model.add(Dense(1,
#                 kernel_initializer="random_uniform",
#                 bias_initializer="ones",
#                 activation="sigmoid"))

model.summary()  # 第一层参数90个是因为输入该层特征元8*本层元数=80，再加上本层8元的8个偏向量为88个

# 步骤三：编译模型

model.compile(loss="binary_crossentropy",  # 损失函数指定交叉熵
              optimizer="adam",  # 优化器指定sgd随机梯度下降法
              # 优化器从 sgd 改为 adam准确度会更好
              metrics=["accuracy"])  # 评估标准一般用accyracy

# 步骤四：训练模型

model.fit(X, Y, epochs=150, batch_size=10, verbose=0)  # 训练150次，每批次尺寸为10，使用verbose=0不显示训练过程

# 步骤五：评估模型

loss, accuracy = model.evaluate(X, Y)
print("准确度={:.2f}".format(accuracy))
