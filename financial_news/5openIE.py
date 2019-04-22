"""
openIE는 로딩하는데 오래걸리니까... 계속 띄워놓을 수 있도록 subprocess로 실행해서 파이프로 연결한 다음에 입출력한다.

기사 하나를 openIE에 집어넣었을 때, extracted line은 한 개가 아니라 여러 개가 나오고 몇 개가 나올지는 모른다.
그래서 한 라인 write하고 한 번 readline하는 식으로는 제대로 extracted된 모든 라인을 가져올 수 없다.
한 라인 write하고 버퍼에 없을 때 까지 지속 readline...하면 다 읽어올 수 있기는 한데 버퍼에 없을 때 blocking에 들어간다.
그래서 그냥 Thread랑 Queue를 써서 해결하는게 더 낫겠다.
"""
import os
import subprocess
import re

from pprint import pprint
import csv

src = open("./4news_translated1-224.csv", 'r', newline='')
dst = open("./5news_IE.csv", 'w', newline='')

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



i = 0
for row in src_reader:
    if i > 1:
        break
    i += 1
    openie.stdin.write(row[2] + "\n")
    openie.stdin.flush()
    while True:
        extracted_str = openie.stdout.readline()[:-1]
        print(extracted_str)
    # if po.search(extracted_str) is None:
    #     # Extracted
    #     print(extracted_str.split("\t"))
    # else:
    #     # Not extracted
    #     pass

src.close()
dst.close()
