"""
openIE는 로딩하는데 오래걸리니까... 계속 띄워놓을 수 있도록 subprocess로 실행해서 파이프로 연결한 다음에 입출력한다.

기사 하나를 openIE에 집어넣었을 때, extracted line은 한 개가 아니라 여러 개가 나오고 몇 개가 나올지는 모른다.
그래서 한 라인 write하고 한 번 readline하는 식으로는 제대로 extracted된 모든 라인을 가져올 수 없다.
한 라인 write하고 버퍼에 없을 때 까지 지속 readline...하면 다 읽어올 수 있기는 한데 버퍼에 없을 때 blocking에 들어간다.
그래서 종료문자열(!@#exit!@#)를 명시적으로 보내서 이게 나올 때 까지 읽는 방식으로 처리.

OpenIE에 쓰는 Thread, OpenIE로부터 읽어오는 Thread를 분리하는 방법을 생각했었는데, csv에서 읽어온 날짜와 시간 정보를 읽는 Thread 쪽에서 알 수가 없다.
이걸 알기 위해서는 동기식으로 동작해야 하는데 이러면 Thread를 써서 얻는 이익이 없어지고, 
비동기식으로 동작하려면 매 line을 읽을 때 마다 Thread를 새로 만들거나, 날짜, 시간 멤버 변수를 세팅해주어야 한다.
물론 비동기식으로 동작하는게 이익은 있을 것 같기는 한데, 그냥 짜도 짧은 시간 내에 작업이 끝날 것 같아서 그냥 짠다.
"""
import os
import subprocess
import re
import threading

from pprint import pprint
import csv


src = open("./4news_translated2019.04.09-2018.01.01_sorted.csv", 'r', newline='', encoding="utf-8",)
dst = open("./5news_IE.csv", 'w', newline='', encoding="utf-8") 

src_reader = csv.reader(src, delimiter=",", quotechar="|")
dst_writer = csv.writer(dst, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)

openie_run_command = 'java -mx4g -cp ./stanford-corenlp-full-2018-10-05/* edu.stanford.nlp.naturalli.OpenIE -threads 4'.split(" ")
openie = subprocess.Popen(openie_run_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

# Wait until OpenIE is loaded.
while True:
    info_msg = openie.stdout.readline()[:-1]
    print(info_msg)
    if "Enter one sentence per line." in info_msg:
        break

po = re.compile(r"^\[main\] INFO edu\.stanford\.nlp\.naturalli\.OpenIE - No extractions in:")

# ot = OutputThread()
# ot.start()

i = 0
exit_str = "!@#exit!@#"
for row in src_reader:
    # if i > 10:
        # break
    i += 1
    openie.stdin.write(row[2] + "\n")
    openie.stdin.write(exit_str + "\n")
    openie.stdin.flush()
    while True:
        extracted_str = openie.stdout.readline()[:-1]
        if po.search(extracted_str) is None:
            # Extracted
            try:
                confidence, subject, relation, _object = extracted_str.split("\t")
                dst_writer.writerow([row[0], row[1], confidence, subject, relation, _object])
            except ValueError as e:
                print(extracted_str.split("\t"))

        else:
            # Not extracted
            if extracted_str[-10:] == exit_str:
                break

src.close()
dst.close()
