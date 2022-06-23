import re
from os import listdir

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

# 扫描文字档清单

path = "R-1 aclImdb/"
fList = [path + "train/pos/" + x for x in listdir(path + "train/pos")] + \
        [path + "train/neg/" + x for x in listdir(path + "train/neg")] + \
        [path + "test/pos/" + x for x in listdir(path + "test/pos")] + \
        [path + "test/neg/" + x for x in listdir(path + "test/neg")]


# 使用正则表达式删除文字中的HTML标签符号

def remove_tags(text):
    tag = re.compile(r'<[^>]+>')
    return tag.sub('', text)


# 建立标签列表（对应pos和neg依序是12500个正面、12500个负面，测试资料也是）

input_label = ([1] * 12500 + [0] * 12500) * 2
input_text = []

# 读取文字档案

for fn in fList:
    with open(fn, encoding="utf8") as ff:
        input_text += [remove_tags("".join(ff.readlines()))]

print(input_text[5])
print(input_label[5])

# Tokenizer处理文档列表

tok = Tokenizer(num_words=2000)  # 字典字数2000个
tok.fit_on_texts(input_text[:25000])  # 处理前25000笔文档
print("文件数： ", tok.document_count)
print({k: tok.word_index[k] for k in list(tok.word_index)[:10]})  # 显示索引字典的前10个

# 使用texts_to_sequences()将文字内容转换为整数清单

X_train = tok.texts_to_sequences(input_text[:25000])
X_test = tok.texts_to_sequences(input_text[25000:])
Y_train = input_label[:25000]
Y_test = input_label[25000:]

# 将序列资料填充成相同长度

X_train = sequence.pad_sequences(X_train, maxlen=100)
X_test = sequence.pad_sequences(X_test, maxlen=100)
print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)
