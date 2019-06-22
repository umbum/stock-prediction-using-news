"""
BERT 모델을 이용해 Sentence Representing을 하는 모듈

labeled_news.reversed.csv 파일이 같은 디렉토리 안에 있어야함
"""

import os
import csv
import pickle
import torch

from flair.data import Sentence
from flair.embeddings import BertEmbeddings, DocumentPoolEmbeddings

print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.is_available())

src = open("./labeled_news.reversed.csv", 'r', newline='', encoding="utf-8")
src_reader = csv.reader(src, delimiter=",", quotechar="|")
dst = open("./4bert_vectors.pickle", "wb")

# init embedding
embedding = BertEmbeddings('bert-base-multilingual-cased')
document_embeddings = DocumentPoolEmbeddings([embedding])


def getBertVector(str):
    # create a sentence
    #print(str)
    sentence = Sentence(str)

    # embed words in sentence
    document_embeddings.embed(sentence)
   
    # print(str)
    # print(sentence.get_embedding().detach().numpy())
    
    return sentence.get_embedding().detach().numpy()


# getBertVector('한·일 관계가 역대 최악으로 치달으면서 관광업계에서도 우려의 목소리가 커지고 있다.')
# getBertVector('한·일 관계가 역대 최악으로 치달으면서 관광업계에서도 우려섞인 목소리가 나오는 중이다.')

i = 0
for row in src_reader:
    i += 1

    try:
        u_vector = getBertVector(row[2])

        pickle_entry = {
            "day": row[0],
            "time": row[1],
            "u-vector": u_vector,
            "label" : row[3]

        }
        pickle.dump(pickle_entry, dst)

        #if i >= 56530:
        #   break;
    except KeyError:
        print("[KeyError] {}".format(row[2]))
        pass
    except RuntimeError as e:
        print(e)
        continue
        

print("{} events were embedded!".format(i))

src.close()



