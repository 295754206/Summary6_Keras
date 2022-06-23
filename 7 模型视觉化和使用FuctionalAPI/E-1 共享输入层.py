from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model

# 定义模型

shared_input = Input(shape=(64, 64, 1))

# 第一个共享输入层的卷积和池化层

conv1 = Conv2D(32, kernel_size=3, activation="relu")(shared_input)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
flat1 = Flatten()(pool1)

# 第二个共享输入层的卷积和池化层

conv2 = Conv2D(16, kernel_size=5, activation="relu")(shared_input)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat2 = Flatten()(pool2)

# 合并2个共享输入层的卷积和池化层

merge = concatenate([flat1, flat2])

# 建立分类器

hidden1 = Dense(10, activation="relu")(merge)
output = Dense(1, activation="sigmoid")(hidden1)
model = Model(inputs=shared_input, outputs=output)

# 输出模型图

plot_model(model, to_file="O-4 model.png", show_shapes=True)