# @author       umbum (umbum7601@gmail.com)
# @date         2019/05/03
# ./5news_IE.csv의 주어, 목적어, 서술어를 각각 벡터로 임베딩한 결과인 ../data/6news_vectors.pickle를 생성하는 스크립트

import os
import csv
from gensim.models import Word2Vec
import gensim.downloader as api
import pickle

### corpus를 사용하는 부분. 매번 학습을 다시 시켜야해서 그냥 full model을 load해서 쓰는게 낫다.
# corpus_name = "text8"
# print("[download|load {} to|from ~/gensim-data/{}]".format(corpus_name, corpus_name))
# corpus = api.load(corpus_name)
# print("[training...]")
# model = Word2Vec(corpus)
# print(model.wv.most_similar("car"))

model_name = "fasttext-wiki-news-subwords-300"
print("[download|load {} to|from ~/gensim-data/{}]".format(model_name, model_name))
model = api.load(model_name)
print(model.most_similar("samsung"))
print(model.most_similar("unveil"))

src = open("../data/news/5news_IE.csv", 'r', newline='', encoding="utf-8")
src_reader = csv.reader(src, delimiter=",", quotechar="|")
dst = open("../data/news/6news_vectors.pickle", "wb")

def getWordVector(word : str):
    word_list = word.split()
    # list 길이가 1이면서 vector space에 존재하는 단어이면 그냥 사용. 아니면 most_similar
    if len(word_list) == 1 and word_list[0] in model.wv:
        return model.wv[word]
    else:
        # 리스트에 포함 된 단어 중 하나라도 vector space에 없는 단어라면 KeyError가 발생할 수 있다.
        return model.wv[model.wv.most_similar(word_list)[0][0]]

i = 0
for row in src_reader:
    i += 1

    try:
        v_subject = getWordVector(row[3])
        v_action = getWordVector(row[4])
        v_object = getWordVector(row[5])

        pickle_entry = {
            "day" : row[0],
            "time" : row[1],
            "subject" : v_subject,
            "action"  : v_action,
            "object"  : v_object
        }
        pickle.dump(pickle_entry, dst)
    except KeyError:
        print("[KeyError] {}\t{}\t{}".format(row[3], row[4], row[5]))
        pass

print("{} events were embedded!".format(i))

src.close()


