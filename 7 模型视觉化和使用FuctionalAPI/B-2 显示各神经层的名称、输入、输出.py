from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(16, kernel_size=(5, 5), padding="same", input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.summary()

# 显示模型各神经层名称、输入张量、输出张量

print("神经层数: ", len(model.layers))
for i in range(len(model.layers)):
    print(i, model.layers[i].name)

print("每一层输入张量: ")
for i in range(len(model.layers)):
    print(i, model.layers[i].input)

print("每一层输出张量: ")
for i in range(len(model.layers)):
    print(i, model.layers[i].output)
