# @author       umbum (umbum7601@gmail.com)
# @date         2019/04/22
# ./3news_preprocessed.csv의 한글 뉴스를 번역한 ./4news_translated{시작일}-{종료일}.csv를 생성하는 스크립트
# google translation api를 사용하므로 이를 사용할 수 있는 환경이어야 한다.

from pprint import pprint
import csv
import re
from google.cloud import translate
import html
import threading
import time

start_date = "2019.12.31"
end_date = "2017.01.01"
src = open("../data/news/3news_preprocessed.csv", 'r', newline='', encoding="utf-8")
dst = open("../data/news/4news_translated.csv", 'w', newline='', encoding="utf-8")

src_lock = threading.Lock()
dst_lock = threading.Lock()

src_reader = csv.reader(src, delimiter=",", quotechar="|")
dst_writer = csv.writer(dst, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)

translate_client = translate.Client()
target = 'en'

class TranslateWorker(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    
    def run(self):
        while True:
            # time.sleep(3)
            with src_lock:
                try:
                    day, _time, statement = next(src_reader)
                    if day > start_date:
                        continue
                    elif day < end_date:
                        raise StopIteration
                    # print(u'Text: {}'.format(statement))
                except StopIteration:
                    break

            translation = translate_client.translate(statement, target_language=target)
            translated_text = html.unescape(translation['translatedText'])
            # print(u'[] Translation: {}'.format(translated_text))
            with dst_lock:
                dst_writer.writerow([day, _time, translated_text])
        


workers = [TranslateWorker() for i in range(50)]
for w in workers:
    w.start()

for w in workers:
    w.join()

src.close()
dst.close()
