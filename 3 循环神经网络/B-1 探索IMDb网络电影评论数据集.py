from tensorflow.keras.datasets import imdb

top_words = 1000  # 取出前1000个常用的单字，不常见的字会被舍弃
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=top_words)

print("X_train.shape: ", X_train.shape)
print("Y_train.shape: ", Y_train.shape)
print("X_test.shape: ", X_test.shape)
print("Y_test.shape: ", Y_test.shape)

print(X_train[0])  # 展示的都是索引值
print(Y_train[0])  # 1是正面评价，0是负面评价

# 解码显示评论内容

max_index = max(max(x) for x in X_train)  # 找出最大索引值：999
print("Max Index: ", max_index)

word_index = imdb.get_word_index()  # 示范找出单词we的索引
example_index = word_index["we"]
print("Example word 'we' index: ", example_index)

decode_word_map = dict([(value, key) for (key, value) in word_index.items()])  # 反转单词索引变成索引单词字典
print(decode_word_map[example_index])

decoded_indices = [decode_word_map.get(i - 3, "?") for i in X_train[0]]  # 真正评论从4开始，需减3，如果找不到就是？
print(decoded_indices)
decoded_review = " ".join(decoded_indices)  # 用空白拼凑成句
print(decoded_review)
