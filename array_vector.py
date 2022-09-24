from collections import Counter
import numpy as np
import nltk
import time

with open("train_deal.txt", mode="r", encoding="UTF-8") as data_train_deal:  # train_negative读取5990条记录
    array_train = data_train_deal.readlines()
with open("test_positive_deal.txt", mode="r", encoding="UTF-8") as data_test_positive_deal:  # test_positive读取2006条记录
    array_test_positive = data_test_positive_deal.readlines()
with open("test_negative_deal.txt", mode="r", encoding="UTF-8") as data_test_negative_deal:  # test_negative读取2004条记录
    array_test_negative = data_test_negative_deal.readlines()

for i in range(0, 5990):  # 转化为2维列表
    array_train[i] = nltk.word_tokenize(array_train[i])
for i in range(0, 2006):  # 转化为2维列表
    array_test_positive[i] = nltk.word_tokenize(array_test_positive[i])
for i in range(0, 2004):  # 转化为2维列表
    array_test_negative[i] = nltk.word_tokenize(array_test_negative[i])

list_count = []  # 统计训练集中出现次数较多的单词
for i in range(0, 5990):
    for j in range(0, len(array_train[i])):
        list_count.append(array_train[i][j])
a = Counter(list_count)
b = sorted(a.items(), key=lambda x: x[1], reverse=True)
c = []
for i in range(0, 2470):  # 特征数目（若修改N此处需修改）
    c.append(b[i][0])

result_train = [[0 for i in range(0, 2470)] for j in range(0, 5990)]  # 统一测试文本和训练文本的向量空间（若修改N此处需修改）
result_test_positive = [[0 for i in range(0, 2470)] for j in range(0, 2006)]  # 统一测试文本和训练文本的向量空间（若修改N此处需修改）
result_test_negative = [[0 for i in range(0, 2470)] for j in range(0, 2004)]  # 统一测试文本和训练文本的向量空间（若修改N此处需修改）

for i in range(0, 5990):  # 布尔权重
    for j in range(0, 2470):  # 若修改N此处需修改
        result_train[i][j] = 1 if c[j] in array_train[i] else 0
for i in range(0, 2006):  # 布尔权重
    for j in range(0, 2470):  # 若修改N此处需修改
        result_test_positive[i][j] = 1 if c[j] in array_test_positive[i] else 0
for i in range(0, 2004):  # 布尔权重
    for j in range(0, 2470):  # 若修改N此处需修改
        result_test_negative[i][j] = 1 if c[j] in array_test_negative[i] else 0

train_x = result_train  # 保存训练文本的向量空间
train_y = [1 for i in range(0, 2995)] + [-1 for i in range(0, 2995)]  # 保存训练文本所属的类

t1 = time.time()

count1 = 0
count2 = 0
k = 8  # knn算法

for p in range(0, 2006):
    distances = []  # 存储距离
    x = result_test_positive[p]
    array1 = np.array(x)
    for i in range(0, 5990):
        array2 = np.array(train_x[i])
        mocheng = np.sqrt(np.sum(array1 ** 2) * np.sum(array2 ** 2))
        xiangliangji = np.sum(array1 * array2)
        d = 1 - xiangliangji / mocheng if mocheng != 0 else 0  # 余弦距离
        distances.append(d)
    distances_sort = sorted(distances)  # 将距离排序
    top = []
    for l in range(0, k):
        distance_short = distances_sort[l]  # 取前k个距离
        index_distance = distances.index(distance_short)  # 检索前k个距离对应的训练语句的位置
        # 取权重为距离分之一
        top.append(train_y[index_distance]/distance_short) if distance_short != 0 else top.append(float("inf"))
    if sum(top) > 0:
        count1 = count1 + 1
    else:
        print("第 %d 个正例判断错误" % (p + 1))

for q in range(0, 2004):
    distances = []  # 存储距离
    x = result_test_negative[q]
    array1 = np.array(x)
    for i in range(0, 5990):
        array2 = np.array(train_x[i])
        mocheng = np.sqrt(np.sum(array1 ** 2) * np.sum(array2 ** 2))
        xiangliangji = np.sum(array1 * array2)
        d = 1 - xiangliangji / mocheng if mocheng != 0 else 0  # 余弦距离
        distances.append(d)
    distances_sort = sorted(distances)  # 将距离排序
    top = []
    for l in range(0, k):
        distance_short = distances_sort[l]  # 取前k个距离
        index_distance = distances.index(distance_short)  # 检索前k个距离对应的训练语句的位置
        # 取权重为距离分之一
        top.append(train_y[index_distance]/distance_short) if distance_short != 0 else top.append(float("-inf"))
    if sum(top) <= 0:
        count2 = count2 + 1
    else:
        print("第 %d 个负例判断错误" % (q + 1))

print("正例的准确率为：", count1/2006)
print("负例的准确率为：", count2/2004)
print("准确率为：", (count1 + count2) / 4010)

t2 = time.time()
print("使用的时间为：", t2 - t1)
