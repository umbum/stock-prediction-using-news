
import sys
import os

def reverseLines(fpath):
    src = open(fpath, 'r')
    fname, ext = os.path.splitext(fpath)
    dst = open("{}.reversed{}".format(fname, ext), 'w')
    i = 0
    for line in src.readlines()[::-1]:
        dst.write(line)
        i += 1
    src.close()
    print("{} lines".format(i))
    

if __name__ == "__main__":
    reverseLines(sys.argv[1])
