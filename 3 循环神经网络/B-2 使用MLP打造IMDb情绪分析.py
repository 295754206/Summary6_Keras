import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence

seed = 10
np.random.seed(seed)

top_words = 1000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=top_words)  # 加载前1000个常用字

# 步骤一、数据预处理

max_words = 500  # 增加每一篇评论内容的字数可以提升识别准确度，这里从100改为500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)  # 裁剪或填充成成规则形状
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

print("X_train.shape: ", X_train.shape)  # (25000,100)
print("X_test.shape: ", X_test.shape)

# 使用Embedding层将评论内容的单字转换为词向量
# 注意Embedding层一定是Sequential模型的第一层
# 参数分别是 input_dim：最大单字数（单字最大数值）、output_dim：输出词向量的维度、input_length:输入值的最大长度
# 若使用 one-hot 则向量是 1000x1000，这里使用Embedding层建立低维度浮点数的紧密连接的矩阵，即 1000x32，1000个字，每个字长度32

model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words)) # Embedding层参数：1000x32

# 步骤二、建立模型

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))

# 步骤三、编译模型

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 步骤四、训练模型

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=5, batch_size=128, verbose=2)

loss, accuracy = model.evaluate(X_test, Y_test)
print("测试资料的准确度：{:.2f}".format(accuracy))
