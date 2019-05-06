from pprint import pprint
import csv
import re
from google.cloud import translate
import html
import threading
import time

start_date = "2017.12.31"
end_date = "2017.01.01"
src = open("./3news_preprocessed.csv", 'r', newline='')
dst = open("./4news_translated{}-{}.csv".format(start_date, end_date), 'w', newline='')

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
