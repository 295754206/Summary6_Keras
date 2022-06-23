from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model

model_input = Input(shape=(784,))
dense1 = Dense(512, activation="relu")(model_input)
dense2 = Dense(128, activation="relu")(dense1)
dense3 = Dense(32, activation="relu")(dense2)

# 第一个分类输出

output = Dense(10, activation="softmax")(dense3)

# 第二个解码器输出

up_dense1 = Dense(128, activation="relu")(dense3)
up_dense2 = Dense(512, activation="relu")(up_dense1)
decoded_outputs = Dense(784)(up_dense2)

# 定义模型

model = Model(model_input, [output, decoded_outputs])

# 输出模型图

plot_model(model, to_file="O-7 model.png", show_shapes=True)
