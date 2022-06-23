from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

model = Sequential()
model.add(Dense(10, input_shape=(8,), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()

plot_model(
    model,
    to_file="O-1 model.png",  # 保存的图片名
    show_shapes=True  # show_shapes是显示输入输出的shape形状
)
