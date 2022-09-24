import nltk.stem
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
with open("train_positive.txt", mode="r", encoding="UTF-8") as data_train_positive:  # 读取train_positive文本
    array_train_positive = data_train_positive.readlines()
with open("train_negative.txt", mode="r", encoding="UTF-8") as data_train_negative:  # 读取train_negative文本
    array_train_negative = data_train_negative.readlines()
array_train = array_train_positive + array_train_negative
with open("test_positive.txt", mode="r", encoding="UTF-8") as data_test_positive:  # 读取test_positive文本
    array_test_positive = data_test_positive.readlines()
with open("test_negative.txt", mode="r", encoding="UTF-8") as data_test_negative:  # 读取test_negative文本
    array_test_negative = data_test_negative.readlines()

for i in range(0, 5990):  # 分词
    array_train[i] = nltk.word_tokenize(array_train[i])
stop_words = stopwords.words("english")  # 去除停用词
for i in range(0, 5990):
    array_train[i] = [w for w in array_train[i] if w not in stop_words]
for i in range(0, 5990):  # 词形还原
    for j in range(0, len(array_train[i])):
        if array_train[i][j] == "n't":
            array_train[i][j] = "not"
        array_train[i][j] = WordNetLemmatizer().lemmatize(array_train[i][j])
for i in range(0, 5990):  # 词干提取
    for j in range(0, len(array_train[i])):
        array_train[i][j] = PorterStemmer().stem(array_train[i][j])
for i in range(0, 5990):  # 去标点
    for j in range(len(array_train[i])-1, -1, -1):
        if not array_train[i][j].isalpha():
            array_train[i].remove(array_train[i][j])

f = open("train_deal.txt", "w", encoding="UTF-8")  # 为便于测试将处理好的文本存入记事本
for i in range(0, 5990):
    for j in range(0, len(array_train[i])):
        f.write(array_train[i][j])
        f.write(" ")
    f.write("\n")
f.close()

for i in range(0, 2006):  # 分词
    array_test_positive[i] = nltk.word_tokenize(array_test_positive[i])
stop_words = stopwords.words("english")  # 去除停用词
for i in range(0, 2006):
    array_test_positive[i] = [w for w in array_test_positive[i] if w not in stop_words]
for i in range(0, 2006):  # 词形还原
    for j in range(0, len(array_test_positive[i])):
        if array_test_positive[i][j] == "n't":
            array_test_positive[i][j] = "not"
        array_test_positive[i][j] = WordNetLemmatizer().lemmatize(array_test_positive[i][j])
for i in range(0, 2006):  # 词干提取
    for j in range(0, len(array_test_positive[i])):
        array_test_positive[i][j] = PorterStemmer().stem(array_test_positive[i][j])
for i in range(0, 2006):  # 去标点
    for j in range(len(array_test_positive[i])-1, -1, -1):
        if not array_test_positive[i][j].isalpha():
            array_test_positive[i].remove(array_test_positive[i][j])

f = open("test_positive_deal.txt", "w", encoding="UTF-8")  # 为便于测试将处理好的文本存入记事本
for i in range(0, 2006):
    for j in range(0, len(array_test_positive[i])):
        f.write(array_test_positive[i][j])
        f.write(" ")
    f.write("\n")
f.close()

for i in range(0, 2004):  # 分词
    array_test_negative[i] = nltk.word_tokenize(array_test_negative[i])
stop_words = stopwords.words("english")  # 去除停用词
for i in range(0, 2004):
    array_test_negative[i] = [w for w in array_test_negative[i] if w not in stop_words]
for i in range(0, 2004):  # 词形还原
    for j in range(0, len(array_test_negative[i])):
        if array_test_negative[i][j] == "n't":
            array_test_negative[i][j] = "not"
        array_test_negative[i][j] = WordNetLemmatizer().lemmatize(array_test_negative[i][j])
for i in range(0, 2004):  # 词干提取
    for j in range(0, len(array_test_negative[i])):
        array_test_negative[i][j] = PorterStemmer().stem(array_test_negative[i][j])
for i in range(0, 2004):  # 去标点
    for j in range(len(array_test_negative[i]) - 1, -1, -1):
        if not array_test_negative[i][j].isalpha():
            array_test_negative[i].remove(array_test_negative[i][j])

f = open("test_negative_deal.txt", "w", encoding="UTF-8")  # 为便于测试将处理好的文本存入记事本
for i in range(0, 2004):
    for j in range(0, len(array_test_negative[i])):
        f.write(array_test_negative[i][j])
        f.write(" ")
    f.write("\n")
f.close()
