# @author       umbum (umbum7601@gmail.com)
# @date         2019/06/09
# BERT로 생성된 바이너리 뉴스 데이터에 주가를 labeling해주는 스크립트

import os
import csv


base_path = os.path.dirname(os.path.abspath(__file__))

def labelingNewsIter(news_file_path, stock_file_path):
    """
    News Labeling Iterator

    Parameters
    ----------
    news_file_path : str
        뉴스 데이터 파일 경로(.csv)
        2019.04.09,11:17, 현대자동차가 올해 글로벌 출시 예정인 엔트리 SUV '베뉴 '의 렌더링 이미지를 9일 최초 공개했다.
    
    stock_file_path : str
        주가 데이터 파일 경로(.csv)

    Usage
    -----
    from data.news_labeling_util_bert import labelingNewsIter

    for day, time, statement, label in labelingNewsIter("./news_preprocessed.reversed.csv", "./stock_reversed.csv"):
        print(day, time, statement, label)
    
    Yields
    ------
    day, time, statement, label
        statement : news 요약 문장
        label : int, 전일 종가 대비 금일 종가가 올랐으면 1, 아니면 0
    """
    news_src = open(news_file_path, 'r', newline='', encoding="utf-8")
    stock_src = open(stock_file_path, 'r', newline='', encoding="utf-8")

    news_src_reader = csv.reader(news_src, delimiter=",", quotechar="|")
    stock_src_reader = csv.reader(stock_src, delimiter=",", quotechar="\"")

    try:
        news_day, time, statement = next(news_src_reader)
        _, pre_closing_price = getStockData(next(stock_src_reader))
        stock_day, closing_price = getStockData(next(stock_src_reader))
        while True:
            if news_day == stock_day:
                yield news_day, time, statement, int(closing_price > pre_closing_price)
                news_day, time, statement = next(news_src_reader)
            elif news_day > stock_day:
                pre_closing_price = closing_price
                stock_day, closing_price = getStockData(next(stock_src_reader))
            elif news_day < stock_day:
                news_day, time, statement = next(news_src_reader)

    except (StopIteration):
        pass
            
    news_src.close()
    stock_src.close()


def getStockData(stock_row):
    """
    day,              closing,   open_price,highest,   lowest,    volume,   rate_of_change
    "2019년 06월 07일","2,072.33","2,070.78","2,081.16","2,057.97","357.53K","0.16%"
    """
    day, closing, open_price, highest, lowest, volume, rate_of_change = stock_row
    return day.replace("년", "").replace("월", "").replace("일", "").replace(" ", "."), closing


if __name__ == "__main__":
    dst = open("./labeled_news.csv", 'w', newline='', encoding="utf-8")
    dst_writer = csv.writer(dst, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)

    for day, time, statement, label in labelingNewsIter("./news_preprocessed.reversed.csv", "./stock_reversed.csv"):
        dst_writer.writerow([day, time, statement, label])
    
    dst.close()