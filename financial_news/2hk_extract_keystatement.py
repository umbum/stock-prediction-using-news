"""
핵심 문장 추출에 gensim을 사용하면 좋겠는데. gensim을 사용하려면 뉴스 전문이 있는게 좋으니까 newspaper4k 같은 걸로 뉴스 전문을 일단 받아야하고.
시간이 이래저래 부족할 것 같아서, 일단은 페이지뷰에서 딸려온 뉴스 앞부분 텍스트에서 첫 번째 문장을 추출하고 이를 핵심 문장으로 간주했다.
"""
from pprint import pprint
import csv


src = open("./1news_korean.csv", 'r', newline='', encoding="utf-8")
dst = open("./2news_keystatement.csv", 'w', newline='', encoding="utf-8")
dst2 = open("./2news_excepted.csv", 'w', newline='', encoding="utf-8")

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
