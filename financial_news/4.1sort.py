# @author       umbum (umbum7601@gmail.com)
# @date         2019/05/03
# 뉴스 csv를 읽어와 (날짜,시간) 순으로 정렬해주는 유틸리티

import os, sys
import csv
from operator import itemgetter

def main(src_fname, dst_fname):
    src_fp = open(src_fname, 'r', newline='')
    dst_fp = open(dst_fname, 'w', newline='')

    src_reader = csv.reader(src_fp, delimiter=",", quotechar="|")
    dst_writer = csv.writer(dst_fp, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
    
    src_lines = []
    for row in src_reader:
        src_lines.append(row)

    src_lines.sort(key=itemgetter(0, 1), reverse=True)

    for row in src_lines:
        dst_writer.writerow(row)

    src_fp.close()
    dst_fp.close()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        src_name, src_ext = os.path.splitext(sys.argv[1])
        dst = src_name + "_sorted" + src_ext
        main(sys.argv[1], dst)
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Usage : python {} <src_file> [<dst_file>]".format(sys.argv[0]))