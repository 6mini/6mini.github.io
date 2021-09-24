---
title: '[Python] Web Scraping'
description: 파이썬을 통한 크롤링, Read HTML or CSS, DOM, requests와 beautifulsoup 라이브러리 사용
categories:
 - Data Engineering
tags: [Data Engineering, Python, Crawling, Web Scraping, DOM, requests, beautifulsoup, 파이썬, 크롤링, 웹스크래핑, 돔]
mathjax: enable
# 0️⃣1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣8️⃣9️⃣🔟
---

# 1️⃣ Reference

- [Beautiful Soup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Requests: HTTP for humans](https://requests.readthedocs.io/en/master/)
- [Python’s Requests Library (Guide)](https://realpython.com/python-requests/)
- [CSS: Cascading Style Sheets](https://developer.mozilla.org/ko/docs/Web/CSS)
- [클래스 선택자](https://developer.mozilla.org/ko/docs/Web/CSS/Class_selectors)
- [ID 선택자](https://developer.mozilla.org/ko/docs/Web/CSS/ID_selectors)

# 2️⃣ HTML & CSS

## HTML
- HTML : HyperText Markup Language : 웹에서 페이지를 표시할 때 사용
- MDN에 의하면 프로그래밍 언어는 아니다.
- 웹 페이지가 어떻게 구성되어 있는지 알려주는 마크업 언어이다.

### Element
- head, body, div, li 등 다양하다.

## CSS
- CSS(Cascading Style Sheets) : 웹 페이지 문서의 표현을 알려주는 스타일 시트 언어

### Selector
- Type selector : CSS 타입에 따라서 선택할 수 있다(예를 들어 'p', 'div' 등).
- Class selector : 클래스에 따라 선택할 수 있다.
- Id selector : id 에 따라 선택할 수 있다.

### 상속
- 요소의 위치에 따라 상위 요소의 스타일을 상속받는다.

```html
<div style="color:red">
    <p>I have no style</p>
</div>
```

### Class
- 특정 요소의 스타일을 정하고 싶을 때 사용된다.

```html
<p class="banana">I have a banana class</p>
```

- '.'을 이용해서 적용한다.

```css
.banana {
    color:"yellow";
}
```

### ID

```html
<p id="pink">My id is pink</p>
```

```css
#pink {
    color:"pink";
}
```

# 3️⃣ DOM
- DOM(Document Object Model) : 문서 객체 모델
- HTML, XML 등 문서의 프로그래밍 인터페이스
- 문서를 하나의 구조화된 형식으로 표현하기 때문에 원하는 동작을 할 수 있다.
- DOM을 통해 프로그래밍 언어에서 사용할 수 있는 데이터 구조 형태로 작업을 수행할 수 있어 크롤링 등 웹 페이지와 작업할 때 매우 중요하다.

## Method
- 웹 브라우저에서 개발자 도구를 열어 콘솔창에서 JS를 통해 사용

```js
document.querySelectorAll('p')
```

### 기능
- `getElementsbyTagName` : 태그 이름으로 문서의 요소를 리턴
- `getElementById` : 'id'가 일치하는 요소를 리턴
- `getElementsByClassName` : '클래스'가 일치하는 요소를 리턴
- `querySelector` : 셀렉터와 일치하는 요소를 리턴
- `querySelectorAll` : 셀렉터와 일치하는 모든 요소를 리턴

# 4️⃣ Web Scraping
- 크롤링과 유사하지만 크롤링은 자동화에 초점이 맞춰져 있다.
- 스크래핑은 특정 정보를 가져오는 것이 목적이라면 크롤링은 인터넷 사이트를 인덱싱하는 목적을 둔다.

## requests 라이브러리

### 설치

```
$ pip install requests
```

### 요청 보내기

```py
import requests
requests.get('https://google.com')
# <Response [200]> : 정상 리턴
```

## BeautifulSoup 라이브러리
- [BeautifulSoup Document](https://urclass.codestates.com/www.crummy.com/software/BeautifulSoup/bs4/doc/)

```
$ pip install beautifulsoup4
```

### Parsing
- Parsing : 문자열로 구성된 특정 문서를 파이썬에서 쉽게 사용할 수 있도록 변환해주는 작업

```py
import requests
from bs4 import BeautifulSoup

url = 'https://google.com'
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')
```

### 요소 찾기

#### `find`, `find_all`

```py
dog_element = soup.find(id='dog')
cat_elements = soup.find_all(class_='cat') # class 를 이용해 찾을 때에는 class 가 아닌 뒤에 밑줄 '_' 을 추가해야 한다.

cat_elements = soup.find_all(class_='cat')

for cat_el in cat_elements: # 결과물 이용 세부 검색
    cat_el.find(class_='fish')
```

#### tag

```py
cat_div_elements = soup.find_all('div', class_='cat')
```

#### string

```py
soup.find_all(string='raining')
soup.find_all(string=lambda text: 'raining' in text.lower()) # 대소문자 구분을 없애 탐색
soup.find_all('h3', string='raining') # 요소로 받기 위해 태그 추가
```

### 정보 얻기

```py
# <p class='cat'>This is a p-cat</p>

cat_el = soup.find('p', class_='cat')
cat_el.text #=> 'This is a p-cat'

cat_el.text.strip() # 불필요한 띄어쓰기 정리
```