from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical

# 指定取出最常见的10000个单字

top_words = 10000
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=top_words)

print("X_train.shape: ", X_train.shape)
print("Y_train.shape: ", Y_train.shape)
print("X_test.shape: ", X_test.shape)
print("Y_test.shape: ", Y_test.shape)

# 显示第1笔资料内容的索引数值清单和对应类别

print(X_train[0])
print(Y_train[0])

# 找到最大索引值

max_index = max(max(sequence) for sequence in X_train)
print("Max Index: ", max_index)  # max_index为9999

# 建立解码字典

word_index = reuters.get_word_index()
we_index = word_index["we"]
print("'we' index: ", we_index)

decode_word_map = dict([(value, key) for (key, value) in word_index.items()])
print(decode_word_map[we_index])

# 解码显示第1笔训练资料集的新闻内容

decoded_indices = [decode_word_map.get(i - 3, "?") for i in X_train[0]]  # 索引前三个值0-2是保留，新闻索引是从4开始需要减3
print(decoded_indices)

decoded_news = " ".join(decoded_indices)
print(decoded_news)

# 将清单转换成张量，以及将清单都填充或裁剪成相同单字数（即长度），即转换为（样本数，最大单字数）

max_words = 200
X_train = sequence.pad_sequences(X_train, maxlen=max_words)  # 统一裁剪成200字长
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)

# One-hot编码

Y_train = to_categorical(Y_train, 46)  # 总共46个主题
Y_test = to_categorical(Y_test, 46)  # 总共46个主题
