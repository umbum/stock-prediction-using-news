"""
preprocessing

1. (.*) 제거
2. [.*] 제거
3. # ◇ =같은 특수 문자 제거도 해야하지만... 이런 뉴스가 많지는 않아서 놔둔다.
"""
from pprint import pprint
import csv
import re


# src = open("./2news_keystatement.csv", 'r', newline='')
# dst = open("./3news_preprocessed.csv", 'w', newline='')

# src_reader = csv.reader(src, delimiter=",", quotechar="|")
# dst_writer = csv.writer(dst, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)


def preprocess(s):
    i = 0
    j = 0
    preprocessed_str = ""
    while (i < len(s)):
        if s[i] == '[':
            preprocessed_str += " " + s[j:max(i, 0)]
            j = i + 1
            while (j < len(s)):
                if s[j] == ']':
                    i = j
                    j += 1
                    break
                j += 1
        elif s[i] == '(':
            preprocessed_str += " " + s[j:max(i, 0)]
            j = i + 1
            while (j < len(s)):
                if s[j] == ')':
                    i = j
                    j += 1
                    break
                j += 1
        elif s[i] == '<':
            preprocessed_str += " " + s[j:max(i, 0)]
            j = i + 1
            while (j < len(s)):
                if s[j] == '>':
                    i = j
                    j += 1
                    break
                j += 1
        i += 1

    preprocessed_str += " " + s[j:]
    return preprocessed_str


for row in src_reader:
    dst_writer.writerow([row[0], row[1], preprocess(row[2])])

src.close()
dst.close()

