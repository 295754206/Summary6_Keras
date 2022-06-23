import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

seed = 10
np.random.seed(seed)

top_words = 10000
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=top_words)

max_words = 200
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
Y_train = to_categorical(Y_train, 46)
Y_test = to_categorical(Y_test, 46)

model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Dropout(0.75))
model.add(LSTM(32, return_sequences=True)) # 第1层LSTM需要指定返回所有序列
model.add(LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(46, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

history = model.fit(
    X_train,
    Y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=32,
    verbose=2
)

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("测试资料的准确度：{:.2f}".format(accuracy))

loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

acc = history.history["accuracy"]
epochs = range(1, len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "b-", label="Training Acc")
plt.plot(epochs, val_acc, "r--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()