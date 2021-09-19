---
title: 'Beautifulsoup Naver Movie Reveiw Web Scraping and Exeport SQLite in Python'
description: 파이썬을 이용하여 네이버 영화에서 리뷰의 한줄평과 점수를 beautifulsoup로 스크래핑하여 리스트로 저장 및 평균 점수를 구하고 SQLite를 통해 DB를 저장하는 과정
categories:
 - Did Unknown
tags: [Did Unknown, Python, beautifulsoup, Naver Movie, Web Scraping, SQLite, 파이썬, 네이버 영화, 스크래핑, 크롤링]
mathjax: enable
# 0️⃣1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣8️⃣9️⃣🔟
---

# 1️⃣ Web Scraping
- 네이버 영화 리뷰 스크래핑 함수 구현
- TEST Movie : 샹치와 텐 링즈의 전설

![스크린샷 2021-09-17 17 25 47](https://user-images.githubusercontent.com/79494088/133750472-1b9b330e-1c78-43ec-acc9-4d4a5406c5c4.png)

```py
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://movie.naver.com/movie"
```

## Page Parsing
- URL을 받아 페이지 가져와서 파싱한 두 결과 리턴

```py
def get_page(page_url):
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup, page
```

## Movie Code

### 검색 페이지 접속

![스크린샷 2021-09-17 17 28 27](https://user-images.githubusercontent.com/79494088/133750927-e73d4c47-5ea9-4188-842a-f1a4c6c5c82b.png)

### 속성 확인
- 검사를 통해 가져와야 할 속성을 확인한다.

![스크린샷 2021-09-17 17 30 08](https://user-images.githubusercontent.com/79494088/133751200-095dc4f4-bc4a-4ae4-a148-5ee81c46f3e7.png)

- 'reult_thumb' class에서 href를 가져와 '='로 split한 뒤 추출한다.

```py
def get_movie_code(movie_title):
    search_url = f"{BASE_URL}/search/result.naver?query={movie_title}&section=all&ie=utf8"
    soup = get_page(search_url)[0]
    movie_code = int(soup.find(class_='result_thumb').find('a')['href'].split('=')[1])
    return movie_code

print(get_movie_code('샹치')) # 187348
```

## Review list 생성

### 페이지 위치
- dictionary 형태

![스크린샷 2021-09-17 17 37 04](https://user-images.githubusercontent.com/79494088/133752178-f4cbda0c-41f7-4b51-9bca-87eaa563012e.png)

- 'title' class를 모두 가져와 for문으로 추출
- text : text화 시켜 '\n'을 기준으로 split한 후 선택
- star : em을 가져와 text만 추출 후 int화

```py
def get_reviews(movie_code, page_num=1):
    review_url = f"{BASE_URL}/point/af/list.naver?st=mcode&sword={movie_code}&target=after&page={page_num}"
    review_list = []
    soup = get_page(review_url)[0] # 위에서 만든 Get page 함수 중 suop만 가져오기
    for i in soup.find_all(class_='title'):
        review_list.append({
            'review_text': i.text.split('\n')[5],
            'review_star': int(i.find('em').text)
        })
    return review_list

print(get_reviews(get_movie_code('샹치')))
'''
[
{'review_text': '', 'review_star': 10},
{'review_text': 'CG와 BMW로 얼룩진 마블판 *** ‘뽕’ 영화 ', 'review_star': 6},
{'review_text': '평이 안좋아서 볼까말까 했으나 결론적으로 난 재밌었다 ', 'review_star': 10},
{'review_text': '감독이 *** 알고 까라 스토리가 별로야 그냥 캡마같아 ', 'review_star': 6},
{'review_text': '***', 'review_star': 7},
{'review_text': '마블이 마블했다. 엄지척. ', 'review_star': 8},
{'review_text': '이렇게 냄새나는 디즈니 영화는 첨이네 ', 'review_star': 1},
{'review_text': '평점이 왜 낮은지 도무지 모르겠습니다.감동적인 영화라 시간가는줄 몰랐습니다.고맙습니다. ', 'review_star': 10},
{'review_text': '재밋는데 왜 난리들이지 *** 걍 싫어하는듯ㅎ ', 'review_star': 9},
{'review_text': '마블 찐팬인데 평가할 가치도없다 ', 'review_star': 1}
]
'''
```

### 리뷰 수
- 영화 이름과 총 스크래핑할 리뷰 수를 받아 해당 수만큼 항목이 담긴 리스트 리턴

```py
def scrape_by_review_num(movie_title, review_num):
    reviews = []
    page_num = 1
    while len(reviews) < review_num:
        reviews += get_reviews(get_movie_code(movie_title), page_num)
        page_num += 1
    return reviews[:review_num]
```

### 페이지 수
- 영화 이름과 총 스크래핑할 페이지 수를 받아 해당 페이지만큼 항목이 담긴 리스트 리턴

```py
def scrape_by_page_num(movie_title, page_num=10):
    reviews = []
    for i in range(page_num):
        reviews += get_reviews(get_movie_code(movie_title), i)
    return reviews
```

## 평균 별점
- 리뷰 리스트를 받아 평균 별점을 구해 리턴

```py
def get_avg_stars(reviews):
    star = []
    for i in reviews:
        star += [i['review_star']]
    avg = sum(star) / len(star)
    return avg

print(get_avg_stars(scrape_by_page_num('샹치'))) # 6.27
```

# 2️⃣ Exeport SQLite
- 영화제목, 페이지 수를 받아 스크래핑한 뒤 DB에 저장

```py
import os
import sqlite3
from src import Web_Scraping

DATABASE_PATH = os.path.join(os.getcwd(), 'scrape_data.db')
conn = sqlite3.connect(DATABASE_PATH)

def store_by_page_num(movie_title, page_num=10, conn=conn):
    cur = conn.cursor()
    Review = Web_Scraping.scrape_by_page_num(movie_title, page_num)
    id = 0
    for row in Review:
        cur.execute(
        "INSERT INTO Review (id, review_text, review_star, movie_title) VALUES (?, ?, ?, ?)",
        (id, row['review_text'], row['review_star'], movie_title)
        )
        id += 1
    conn.commit()

def init_db(conn=conn): # Review 테이블 초기화 함수
    create_table = """CREATE TABLE Review (
                        id INTEGER,
                        review_text TEXT,
                        review_star FLOAT,
                        movie_title VARCHAR(128),
                        PRIMARY KEY (id)
                        );"""

    drop_table_if_exists = "DROP TABLE IF EXISTS Review;"
    cur = conn.cursor()
    cur.execute(drop_table_if_exists)
    cur.execute(create_table)
    cur.close()
```