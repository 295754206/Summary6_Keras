from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer

# 一、分割文字资料

doc = "Keras is an API designed for human beings, not machines."
words = text_to_word_sequence(doc)
print(words)

doc = "Apple, Samsung, Huawei, Xiaomi"
words = text_to_word_sequence(
    doc,  # 第一个是text参数，第二个filter参数指定过滤掉哪些字元，预设包括','等标点符号
    lower=False,  # lower参数默认是True自动改小写
    split=","  # split参数默认是空白
)
print(words)

# 二、计算文字资料的字数

doc = "This is a book, that is a pen"
words = set(text_to_word_sequence(doc))
vocab_size = len(words)
print(words)
print(vocab_size)

# 三、显示文字资料的摘要资讯

docs = [
    "Keras is an API designed for human beings, not machines.",
    "Easy to learn and easy to use.",
    "Keras makes it easy to turn models into products."
]

tok = Tokenizer()
tok.fit_on_texts(docs)

print(tok.document_count)  # 显示有几笔资料
print(tok.word_counts)  # 显示每一个单字出现的次数
print(tok.word_docs)  # 显示各个单字在几份文件中出现
print(tok.word_index)  # 显示单字索引

words = tok.texts_to_sequences(docs)  # 显示文字资料索引化
print(words)
