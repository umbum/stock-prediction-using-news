"""
thread pool 구성해서 쭉 돌려야... 그냥 리퀘스트하면 너무 오래걸려.
"""
from pprint import pprint
import csv
import re
from google.cloud import translate
import html

# src = open("./3news_preprocessed.csv", 'r', newline='')
# dst = open("./4news_translated.csv", 'w', newline='')

# src_reader = csv.reader(src, delimiter=",", quotechar="|")
# dst_writer = csv.writer(dst, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)

translate_client = translate.Client()
target = 'en'

i = 0
for row in src_reader:
    i += 1
    if i < 223:
        continue
    translation = translate_client.translate(row[2], target_language=target)
    translated_text = html.unescape(translation['translatedText'])

    # print(u'Text: {}'.format(row[2]))
    print(u'[{}] Translation: {}'.format(i, translated_text))
    with lock:
        dst_writer.writerow([row[0], row[1], translated_text])  



with ThreadPoolExecutor(max_workers=20) as e:
    for i in range(1, 1):
        e.submit(requestAndWrite, i)

src.close()
dst.close()
