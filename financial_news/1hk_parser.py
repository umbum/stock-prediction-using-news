
from bs4 import BeautifulSoup
import csv
import os
from natsort import natsorted

# find_all day_article해서 for문 돌면서 day_article을 다 돌 때 까지.(어차피 하나거나 두개겠지만. 개념상.)
# 날짜 : 04.082019    -> 2019.04.08
# soup.find_all(class_="day_article")[0].find(class_="day").get_text()

# 개별 기사 1개씩.
# soup.find_all(class_="day_article")[0].find(class_="article_list").find_all(class_="txt")

# 시간 : '15:24'
# soup.find_all(class_="day_article")[0].find(class_="article_list").find_all(class_="txt")[0].find(class_="time").get_text()

# 타이틀 : "조양호 '폐질환' 왜 공개하지 않았나…병 숨기고 미국서 치료"
# soup.find_all(class_="day_article")[0].find(class_="article_list").find_all(class_="txt")[0].find(class_="tit").get_text()

# 요약문 : '폐섬유종 가능성…"여론 악화로 \'질병 핑계 삼는다\'는 비판 우려한듯"  "3월말 사내이사 연임 실패후 스트레스 등으로 급속 병세악화"  조양호(70) 한진그룹 회장이 8일 갑작스럽게 세상을 떠나면서 그의 사망 원인에도 관심이 쏠린다.  한진그룹은 이날 조 회장 별세 소식을 알리면서 "조 회장이 오늘 새벽 미국 현지에서 폐질환으로 별세했다"고 밝혔다.  조 회장의 충...'
# soup.find_all(class_="day_article")[0].find(class_="article_list").find_all(class_="txt")[0].find(class_="read").get_text()

# csv format : 2019.04.08,15:24,타이틀,요약문

fnames = os.listdir("../htmls")
fnames = natsorted(fnames)

csv_file = open("1news_korean.csv", "w", newline='')
news_writer = csv.writer(csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)

for fname in fnames:
    print(fname)
    with open("../htmls/{}".format(fname), "r") as f:
        html_str = f.read()

    soup = BeautifulSoup(html_str, "html.parser")
    # print(soup.prettify())

    days_articles = soup.find_all(class_="day_article")
    for day_articles in days_articles:
        _day = day_articles.find(class_="day").get_text()
        day = _day[5:] + "." + _day[:5]    # 2019.04.08
        articles = day_articles.find(class_="article_list").find_all(class_="txt")
        for article in articles:
            time = article.find(class_="time").get_text()
            title = article.find(class_="tit").get_text()
            summary = article.find(class_="read").get_text()
            
            news_writer.writerow([day, time, title, summary])

csv_file.close()
