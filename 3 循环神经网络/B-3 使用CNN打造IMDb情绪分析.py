import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, MaxPooling1D, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence

seed = 10
np.random.seed(seed)

# 步骤一、数据载入和预处理

top_words = 1000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=top_words)

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# 步骤二、建立模型

model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Dropout(0.25))
model.add(Conv1D(filters=32,  # 注意这里时间序列是1D不是2D，参数分别是 过滤器数、过滤器尺寸等
                 kernel_size=3,
                 padding="same",
                 activation="relu"))  # Conv1D层参数：32x(3)x32+32 Embedding层输出通道32x过滤器窗口大小x过滤器数+过滤器数的偏向量
model.add(MaxPooling1D(pool_size=2))  # 比例缩小一半
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))

# 步骤三、编译模型

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 步骤四、训练模型

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=5, batch_size=128, verbose=2)

# 测试

loss, accuracy = model.evaluate(X_test, Y_test)
print("测试资料集的准确度：{:.2f}".format(accuracy))
