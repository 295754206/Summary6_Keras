from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model

# 第一个灰阶图片输入

input1 = Input(shape=(28, 28, 1))
conv11 = Conv2D(16, (3, 3), activation="relu")(input1)
pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
conv12 = Conv2D(32, (3, 3), activation="relu")(pool11)
pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
flat1 = Flatten()(pool12)

# 第二个彩色图片输入

input2 = Input(shape=(28, 28, 3))
conv21 = Conv2D(16, (3, 3), activation="relu")(input2)
pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
conv22 = Conv2D(32, (3, 3), activation="relu")(pool21)
pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
flat2 = Flatten()(pool22)

# 合并2个输入

merge = concatenate([flat1, flat2])
dense1 = Dense(512, activation="relu")(merge)
dense2 = Dense(128, activation="relu")(dense1)
dense3 = Dense(32, activation="relu")(dense2)
output = Dense(10, activation="softmax")(dense3)

# 定义多输入模型

model = Model(inputs=[input1, input2], outputs=output)

# 输入图像

plot_model(model, to_file="O-6 model.png", show_shapes=True)
