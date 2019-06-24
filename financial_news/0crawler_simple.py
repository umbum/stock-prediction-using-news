# @author       umbum (umbum7601@gmail.com)
# @date         2019/04/08
# 뉴스 데이터를 크롤링하는 스크립트

import sys
import requests
from concurrent.futures import ThreadPoolExecutor


def requestAndWrite(i):
    URL = "https://www.hankyung.com/all-news/economy?page={}"
    # URL = "https://localhost/all-news/economy?page={}"

    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
        "Cookie": "_ga=GA1.2.1099645471.1554651094; _gid=GA1.2.596745316.1554651096; __gads=ID=d8d44272944cd349:T=1554651095:S=ALNI_MasKS3bTG_WTYwv-R2T5Y4KebKyQQ; gtmdlkr=",
        "Host": "www.hankyung.com",
        "Referer": "https://www.hankyung.com/all-news/economy?page=",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36",
    }
    headers["Referer"] = URL.format(i-1)
    r = requests.get(URL.format(i), headers=headers)
    if len(r.text) < 10000:
        raise Exception
    with open("../data/news/htmls/{}.html".format(i), 'w') as f:
        f.write(r.text)
    print("done {}".format(i))


if __name__ == "__main__":
    if (len(sys.argv) == 3):
        start_page = sys.argv[1]
        end_page = sys.argv[2]
    else:
        print("Usage : python {} <start_page> <end_page>".format(__file__))
        sys.exit()

    with ThreadPoolExecutor(max_workers=20) as e:
        for i in range(int(start_page), int(end_page)):
            e.submit(requestAndWrite, i)