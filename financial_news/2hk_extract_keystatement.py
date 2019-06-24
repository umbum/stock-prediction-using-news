# @author       umbum (umbum7601@gmail.com)
# @date         2019/04/22
# ./1news_korean.csv의 각 row에서 핵심 문장을 추출한 결과인 ./2news_keystatement.csv를 생성하는 스크립트

from pprint import pprint
import csv


src = open("../data/news/1news_korean.csv", 'r', newline='', encoding="utf-8")
dst = open("../data/news/2news_keystatement.csv", 'w', newline='', encoding="utf-8")
dst2 = open("../data/news/2news_excepted.csv", 'w', newline='', encoding="utf-8")

src_reader = csv.reader(src, delimiter=",", quotechar="|")
dst_writer = csv.writer(dst, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
dst2_writer = csv.writer(dst2, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)

hit_count = 0
i = 0
for row in src_reader:
    i += 1
    key_statement = None
    for s in row[3].split("  "):
        if "다." in s:
            key_statement = s[:s.find("다.") + 2]
            break
    
    if key_statement is not None:
        hit_count += 1
        dst_writer.writerow([row[0], row[1], key_statement])
    else:
        dst_writer.writerow([row[0], row[1], row[2]]) # 핵심 문장이 발견되지 않으면 그냥 title을 사용한다.
        dst2_writer.writerow([row[0], row[1], row[2], row[3]]) # 핵심 문장이 발견되지 않으면 그냥 title을 사용한다.
        pass

print("total : {} / preprocessed : {} / unextracted : {}".format(i, hit_count, i - hit_count))
src.close()
dst.close()
dst2.close()
