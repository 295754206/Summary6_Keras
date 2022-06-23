import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence

seed = 10
np.random.seed(seed)

# 步骤一、资料载入和预处理

top_words = 1000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=top_words)

max_words = 100
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# 步骤二、定义模型

model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Dropout(0.25))
model.add(GRU(32, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))

# 步骤三、编译模型

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 步骤四、训练模型

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=5, batch_size=128, verbose=2)

# 评估模型、绘图

loss, accuracy = model.evaluate(X_test, Y_test)
print("测试资料集的准确度：{:.2f}".format(accuracy))

loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

accuracy = history.history["accuracy"]
epochs = range(1, len(accuracy) + 1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, accuracy, "b-", label="Training Acc")
plt.plot(epochs, val_acc, "r--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
