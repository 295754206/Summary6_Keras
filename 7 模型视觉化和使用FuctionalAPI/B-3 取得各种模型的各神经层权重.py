from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model


# 取得权重和偏向量形状函数

def show_weights(model):
    for i in range(len(model.layers)):
        print("神经层:", i, "  名称:", model.layers[i].name)
        weights = model.layers[i].get_weights()
        for j in range(len(weights)):
            print("  ==> ", j, weights[j].shape)  # 0是权重形状，1是偏向量形状


# 取得MLP各神经层权重

model = Sequential()
model = load_model("R-2 titanic.h5")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

show_weights(model)
print()

# 取得CNN各神经层权重

model = load_model("R-3 mnist.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

show_weights(model)
print()

# 取得RNN神经层权重

model = load_model("R-4 imdb_rnn.h5")
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

show_weights(model)
print()

# 取得LSTM神经层权重

model = load_model("R-5 imdb_lstm.h5")
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

show_weights(model)
print()

# 取得GRU神经层权重

model = load_model("R-6 imdb_gru.h5")
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

show_weights(model)
print()
