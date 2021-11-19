---
title: '[Computer Science] Python Programming'
description: 파이썬 메소드에 대해 내부로직에 대한 생각과 각종 상황에서의 활용, 다양한 코드를 활용한 컬렉션 자료형(리스트, 튜플, 셋, 딕셔너리), 현실상황을 수학적 기초개념(사칙연산)으로 프로그래밍
categories:
 - Computer Science
tags: [Computer Science, Python, Lambda, re]
mathjax: enable
---

- 방향: 문제 해결을 위해 파이썬을 활용하고, 함수, 반복과 조건개념에 익숙해지는 것
- CS의 최종 목적: 자료구조와 알고리즘을 이해하며 프로그래밍 하는 것
- CS의 핵심 키워드: 문제 해결과 컴퓨팅 사고력

- [데이터 과학자가 미래에 핫한 직업인 이유](https://youtu.be/dZZfDj_ieEU)
- [프로그래밍을 배워야 하는 이유는?](https://www.youtube.com/watch?v=SESuctdE9vM)
- (둘다 봤던거누ㅋㅋㅋ, 문득 결국 이런 것들이 모여 내가 개발을 시작하게 된 것인가 싶네)

# Programming

## 문제해결
- 복잡한 문제를 작은 문제로 분할하면서 해결한다.
- 문제에 대한 패턴을 발견한다.
- 최소한의 비용으로 최대한 빠르게 해결한다.

### 참고
- [문제 해결을 위한 과학적 사고](https://dojang.io/mod/page/view.php?id=2151)
- [소프트웨어 교육과 파이썬](https://www.youtube.com/watch?v=DZSde316k3E)

## 기반기술

<img width="600" alt="스크린샷 2021-11-17 15 49 33" src="https://user-images.githubusercontent.com/79494088/142148754-79fd7ac3-5e31-4b47-bcbd-2de4033df6f8.png">

- 파이썬, 알고리즘, 자료구조는 생산을 위한 도구이다.
- 파이썬: 컴퓨터와의 소통언어(ex. 수학)
- 알고리즘: 효율적인 문제해결방법(ex. 사칙연산 or 미적분)
- 자료구조: 프로그램의 구조와 크기(ex. 수학문제들간의 관계와 난이도)

# Control Statement

## 정규표현식
- 특정한 규칙을 가진 문자열의 집합을 표현하는 형식이다.
- 복잡한 문자열을 처리할 때 사용하는 기법이다.
- 파이썬 이외에도 모든 프로그래밍 언어에서 공통적으로 사용한다.

```py
import re

wordlist = ["color", "colour", "work", "working",
            "fox", "worker", "working"]

for word in wordlist:
    if re.search('col.r', word) : 
        print (word)
'''
color
'''


regular_expressions = '<html><head><title>Title</title>'
print(len(regular_expressions))

print(re.match('<.*>', regular_expressions).span())

print(re.match('<.*>', regular_expressions).group())
'''
32
(0, 32)
<html><head><title>Title</title>
'''


# case 1_1
phone = re.compile(r"""
010-    # 핸드폰 앞자리 
\d{4}-  # 중간자리
\d{4}  # 뒷자리
""", re.VERBOSE)


phone = re.compile(r"010-\d{4}-\d{4}")


# case 1_3
info = ['홍길동 010-1234-1234', '고길동 010-5678-5679']

for text in info:
    match_object = phone.search(text)
    print(match_object.group())
'''
010-1234-1234
010-5678-5679
'''
```

- [정규 표현식](https://ko.wikipedia.org/wiki/%EC%A0%95%EA%B7%9C_%ED%91%9C%ED%98%84%EC%8B%9D)
- [정규 표현식 살펴보기](https://wikidocs.net/1642)

## 다양한 메소드

### rjust(width, [fillchar])
- 원하는 문자를 따로 지정하고, 다른 문자열로 앞 부분을 채워줄 수 있다.

```py
print("2".rjust(3,"0")) # "002"
 
print("50000".rjust(5,"0")) # "50000"
 
print("123".rjust(5,"0")) # "00123"
 
print("123".rjust(5,"a")) # "aa123"
```

### zfill(width)

```py
print("2".zfill(3)) # "002"
 
print("50000".zfill(5)) # "50000"
 
print("123".zfill(5)) # "00123"
```

### 얕은 복사(copy())

```py
fruits = {"apple", "banana", "cherry"}
fruits_copy = fruits.copy()
fruits_copy
'''
{'apple', 'banana', 'cherry'}
'''


a = {'a': 5, 'b': 4, 'c': 8}
b = a
del b['a']
print(b)
print(a)
'''
{'b': 4, 'c': 8}
{'b': 4, 'c': 8}
'''


import copy
a = {'a': 5, 'b': 4, 'c': 8}
b = copy.copy(a)
del b['a']
print(b)
print(a)
'''
{'b': 4, 'c': 8}
{'a': 5, 'b': 4, 'c': 8}
'''
```

### 깊은 복사(deepcopy())
- 깊은 복사는 내부에 객체들까지 새롭게 copy되는 것이다.
- 완전히 새로운 변수를 만드는 것이다.

```py
import copy
list_var = [[1,2],[3,4]]
list_var_deepcopy = copy.deepcopy(list_var)
list_var_copy = list_var.copy()

list_var[1].append(5)

print(list_var)  # 원래 변수

print(list_var_deepcopy)  # deepcopy : append와 같은 메소드를 써도 값이 변경되지 않음

print(list_var_copy)  # copy : 원본이 변경되었으므로 함께 변경됨
'''
[[1, 2], [3, 4, 5]]
[[1, 2], [3, 4]]
[[1, 2], [3, 4, 5]]
'''
```

- [얕은 복사, 깊은 복사 파헤치기](https://blueshw.github.io/2016/01/20/shallow-copy-deep-copy/)

## 반복문과 조건문

### zip

```py
# zip 함수 활용
a = [1,2,3,4,5]
b = [10,20,30,40,50]
c = zip(a,b)
print(list(c))
'''
[(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
'''


# 반복문과 zip 활용
a = [1,2,3,4,5]
b = [10,20,30,40,50]
c = [100,200,300,400,500]
for x,y,z in zip(a,b,c):
   print(x,y,z)
'''
1 10 100
2 20 200
3 30 300
4 40 400
5 50 500
'''
```

## 에러상황파악

```py
# 케이스 1 - IndentationError
def print_list(list):
for item in list: # 에러 확인 및 해결필요
print(item)


# 케이스 2 - SyntaxError
123ddf


# 케이스 3 - KeyboardInterrupt
while True:
  pass


# 케이스 4 - TypeError
print(1) / 232

a,b = 0
print(a,b)


# 케이스 5 - ZeroDivisionError
value = 2/0


# 특이 케이스 6 - 경고(warning)
def A():
  a = 0
  c =0
  print(a,c,) # 경고는 명시적으로 보이지 않지만, 메모리 비효율/휴먼 에러 등이 발생할 수 있다.
```

## Collection 자료형

### 내장 메소드

#### append(), extend(), insert()
    - a.insert(len(a), x) 는 a.append(x) 와 동등합니다.
- [파이썬 자료 구조](https://urclass.codestates.com/2cc42947-2a15-4b11-97ec-ddc4b512183a?playlist=592#:~:text=append()%2C%20extend()%2C%20insert,%ED%8C%8C%EC%9D%B4%EC%8D%AC%20%EC%9E%90%EB%A3%8C%20%EA%B5%AC%EC%A1%B0)

```py
my_list=[]
for i in range(1000, 2200):
    if (i%7==0) and (i%5!=0):
        my_list.append(str(i))

print(','.join(my_list))
'''
1001,1008,1022,1029,1036,1043,1057,1064,1071,1078,1092,1099,1106,1113,1127,1134,1141,1148,1162,1169,1176,1183,1197,1204,1211,1218,1232,1239,1246,1253,1267,1274,1281,1288,1302,1309,1316,1323,1337,1344,1351,1358,1372,1379,1386,1393,1407,1414,1421,1428,1442,1449,1456,1463,1477,1484,1491,1498,1512,1519,1526,1533,1547,1554,1561,1568,1582,1589,1596,1603,1617,1624,1631,1638,1652,1659,1666,1673,1687,1694,1701,1708,1722,1729,1736,1743,1757,1764,1771,1778,1792,1799,1806,1813,1827,1834,1841,1848,1862,1869,1876,1883,1897,1904,1911,1918,1932,1939,1946,1953,1967,1974,1981,1988,2002,2009,2016,2023,2037,2044,2051,2058,2072,2079,2086,2093,2107,2114,2121,2128,2142,2149,2156,2163,2177,2184,2191,2198
'''


my_list=[]
for i in range(1000, 2200):
    if (i%7==0) and (i%5!=0):
        my_list.insert(len(my_list), str(i))

print(','.join(my_list))
'''
1001,1008,1022,1029,1036,1043,1057,1064,1071,1078,1092,1099,1106,1113,1127,1134,1141,1148,1162,1169,1176,1183,1197,1204,1211,1218,1232,1239,1246,1253,1267,1274,1281,1288,1302,1309,1316,1323,1337,1344,1351,1358,1372,1379,1386,1393,1407,1414,1421,1428,1442,1449,1456,1463,1477,1484,1491,1498,1512,1519,1526,1533,1547,1554,1561,1568,1582,1589,1596,1603,1617,1624,1631,1638,1652,1659,1666,1673,1687,1694,1701,1708,1722,1729,1736,1743,1757,1764,1771,1778,1792,1799,1806,1813,1827,1834,1841,1848,1862,1869,1876,1883,1897,1904,1911,1918,1932,1939,1946,1953,1967,1974,1981,1988,2002,2009,2016,2023,2037,2044,2051,2058,2072,2079,2086,2093,2107,2114,2121,2128,2142,2149,2156,2163,2177,2184,2191,2198
'''


values = []
for i in range(100, 300):
    char = str(i)

    if (int(char[0])%2==0) and (int(char[2])%2==0):
        values.append(char)

print(",".join(values))
'''
200,202,204,206,208,210,212,214,216,218,220,222,224,226,228,230,232,234,236,238,240,242,244,246,248,250,252,254,256,258,260,262,264,266,268,270,272,274,276,278,280,282,284,286,288,290,292,294,296,298
'''


list_1 = ['bread', 'meat']
list_2 = ['Lettuce',2 ,5]
list_1.extend(list_2)
print('list1: {}, list2: {}'.format(list_1, list_2))
'''
list1: ['bread', 'meat', 'Lettuce', 2, 5], list2: ['Lettuce', 2, 5]
'''
```

#### del, remove, pop

```py
list1 = [11, 12, 43, 4, 6]
for i in list1.copy():
    if not i % 2:
        list1.remove(i)
print(list1)
'''
[11, 43]
'''


my_list = [1, 2, 3, 4, 5]
my_list[0] = 99
print(my_list)

del my_list[0]
print(my_list)
'''
[99, 2, 3, 4, 5]
[2, 3, 4, 5]
'''


my_list = [1, 2, 3, 4, 5]
my_list[0] = 99
print(my_list)

my_list.pop()
print(my_list)
'''
[99, 2, 3, 4, 5]
[99, 2, 3, 4]
'''
```

#### count()와 index()
- [파이썬의 list.count() vs list.index() 메소드 특징](https://urclass.codestates.com/2cc42947-2a15-4b11-97ec-ddc4b512183a?playlist=592#:~:text=count()%EC%99%80%20index,%EB%A9%94%EC%86%8C%EB%93%9C%20%ED%8A%B9%EC%A7%95%20%EC%82%B4%ED%8E%B4%EB%B3%B4%EA%B8%B0)

```py
my_list = ['xyz', 'XYZ' 'abc', 'ABC']
print("Index for xyz : ",  my_list.index( 'xyz' ))
print("Index for ABC : ",  my_list.index( 'ABC' ))
'''
Index for xyz :  0
Index for ABC :  2
'''


my_list = ['xyz', 'XYZ' 'abc', 'ABC']
print("Count for xyz : ",  my_list.count( 'xyz' ))
print("Count for ABC : ",  my_list.count( 'ABC' ))
'''
Count for xyz :  1
Count for ABC :  1
'''


import math
def bin_search(li, element):
    bottom = 0
    top = len(li)-1
    index = -1
    while top >= bottom and index == -1:
        mid = int(math.floor((top+bottom) / 2.0))
        if li[mid] == element:
            index = mid
        elif li[mid] > element:
            top = mid-1
        else:
            bottom = mid+1

    return index

li=[2,5,7,9,11,17,222]
print(bin_search(li,11))
print(bin_search(li,102))
'''
4
-1
'''


li = [12,24,35,70,88,120]
for (i,x) in enumerate(li):
   if i not in (0,3,5):
     li = x
print(li)
'''
88
'''


def list_update(data):
    new_li=[]
    new_set = set()
    for item in data:
        if item not in new_set:
            new_set.add(item)
            new_li.append(item)

    return new_li

list_test=[120,120,10,20,30,20]
print(list_update(list_test))
'''
[120, 10, 20, 30]
'''
```

# Lambda
- [람다 대수](https://ko.wikipedia.org/wiki/%EB%9E%8C%EB%8B%A4_%EB%8C%80%EC%88%98)
    - 함수는 컴퓨터 과학과 수학의 기초를 이루는 개념이다.
    - 람다 대수는 함수를 단순하게 표현할 수 있도록 하여 '함수의 계산'이라는 개념을 더 깊이 이해할 수 있게 돕는다.
- 람다는 인라인으로 작성할 수 있기 때문에 전체 함수보다 읽기 쉽다. 따라서 함수 표현식의 규모가 작을 때 람다를 사용하는 것이 좋다.
- 람다 함수의 장점은 함수 객체를 반환한다. 따라서 함수 객체를 인수로 필요로하는 map 또는 filter와 같은 함수와 함께 사용할 때 유용하다.

```py
# 함수정의
define_word = (lambda word1,define :  word1 * define)

# 함수호출
result = define_word('call_result_',5)

# 결과출력
print(result)
'''
call_result_call_result_call_result_call_result_call_result_
'''


# 리스트 생성
spelling = ["test1", "test2", "test4 test5", "test3"]

# 람다함수적용
shout_spells = map(lambda item: item + ('!!!'), spelling)

# 리스트형태로 변환
shout_spells_list = list(shout_spells)

# 결과출력
print(shout_spells_list)
'''
['test1!!!', 'test2!!!', 'test4 test5!!!', 'test3!!!']
'''


# 리스트 생성
fellowship = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']

# 람다함수적용
result = filter(lambda member: len(member) > 6, fellowship)

# 리스트형태로 변환
result_list = list(result)

# 결과출력
print(result_list)
'''
['samwise', 'aragorn', 'boromir', 'legolas', 'gandalf']
'''


# functools 모듈 사용
from functools import reduce

# 리스트 생성
stark = ['robb', 'sansa', 'arya', 'brandon', 'rickon']

# 람다함수적용
result = reduce(lambda item1, item2:  item1+item2, stark)

# 결과출력
print(result)
'''
robbsansaaryabrandonrickon
'''
```

# Review

## TIL
- 파이썬의 다양한 활용법
- 파이썬 메소드의 활용
- 파이썬 컬렉션 자료형의 활용 

## TIWL
- 수학기본개념이 들어간 파이썬 코드를 다양하게 활용해보기
- 컬렉션 자료형에 대해 생각해보기

# Reference
- [파이썬 공식문서 : 리스트 슬라이싱]()
- 프로그래밍과 함께 생각해보기
    - [if문 다루기](https://www.acmicpc.net/step/4)
    - [for문 다루기](https://www.acmicpc.net/step/3)
    - [문자열 다루기](https://www.acmicpc.net/step/7)