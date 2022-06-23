from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model

# 定义模型

model_input = Input(shape=(100, 1))
lstm = LSTM(32)(model_input)

# 第一个共享特征提取层的解释层

extract1 = Dense(16, activation="relu")(lstm)

# 第二个共享特征提取层的解释层

dense1 = Dense(16, activation="relu")(lstm)
dense2 = Dense(32, activation="relu")(dense1)
extract2 = Dense(16, activation='relu')(dense2)

# 合并享特征提取层的解释层

merge = concatenate([extract1, extract2])

# 输出层

output = Dense(1, activation="sigmoid")(merge)
model = Model(inputs=model_input, outputs=output)

# 输出模型图

plot_model(model, to_file="O-5 model.png", show_shapes=True)
