---
title: '[스파크] 트랜스포메이션(Transformations)과 액션(Actions)'
description: 스파크(Spark)의 트랜스포메이션(Narrow, Wide)과 액션에 대한 개념과 함수, 코드 예제
categories:
 - Data Engineering
tags: [스파크, 트랜스포메이션, 액션, 데이터 엔지니어링]
---

- 스파크의 두 가지 오퍼레이션인 트랜스포메이션(Transformations)과 액션(Actions)을 코드를 통해 알아볼 것이다.

# 트랜스포메이션(Transformations)
- 결과값으로 새로운 RDD를 반환한다.
- 지연 실행(Lazy Execution)이 된다는 특징이 있다.
- 트랜스포메이션은 네로우(Narrow)와 와이드(Wide)로 나눌 수 있다.

## 네로우(Narrow) 트랜스포메이션

<img width="535" alt="image" src="https://user-images.githubusercontent.com/79494088/145677992-7b118d94-680a-4c5e-b54b-f45402c5f51d.png">

- 1대 1로 변환된다는 특정이 있다.
- 대표적으로 `filter()`, `map()`, `flatMap()`, `sample()`, `union()` 등이 있다.
- 1열을 조작하기 위해 다른 열과 파티션의 데이터를 끌어다 쓸 필요가 없다.
- 정렬 같은 경우 새로운 RDD를 만들 때 다른 열을 참고해야 하는데 이는 많은 통신이 일어나며, 그렇기에 네로우 트랜스포메이션은 빠르다고 볼 수 있다.

## 와이드(Wide) 트랜스포메이션

<img width="541" alt="image" src="https://user-images.githubusercontent.com/79494088/145678219-26a653f0-240f-438f-bb16-65915c5e216a.png">

- 셔플링을 한다는 특징이 있다.
- 대표적인 예로 `intersection` and `join`, `distinct`, `cartesial`, `reduceByKey()`, `groupByKey()` 등의 함수가 있다.
- 아웃풋 RDD의 파티션에 다른 파티션의 데이터가 들어갈 수 있다.
- 데이터가 나뉘어진 파티션을 왔다 갔다하면, 통신이 많이 필요하기 때문에 성능 상 많은 리소스를 요구한다. 이를 최소화하거나 최적화하여야 성능을 끌어내기 좋다.

## 함수
- `map()`
- `flatMap()`
- `filter()`
- `distinct()`
- `reduceByKey()`
- `groupByKey()`
- `mapValues()`
- `flatMapValues()`
- `sortByKey()`

## 코드
- 프로그램이 트랜스포메이션을 만날 때 새로운 알디디를 만들지 않고 액션을 만나야 연산을 시작하면서 만들어진다.
- 트랜스포메이션을 만나게 되면 신텍스를 체크 한 다음에 기억을 하고 넘어간다. 
- 트랜스포메이션을 여러번 거치면 순서대로 변환 과정을 기억해놓으며 하나의 DAGs를 만들게 된다.

```py
sc.parallelize([1, 2, 3]).map(lambda x: x + 2).collect() # 새로운 RDD 생성
'''
[3, 4, 5]
'''


movies = [
    "그린 북",
    "매트릭스",
    "토이 스토리",
    "캐스트 어웨이",
    "포드 V 페라리",
    "보헤미안 랩소디",
    "빽 투 더 퓨처",
    "반지의 제왕",
    "죽은 시인의 사회"
]

movies_rdd = sc.parallelize(movies)
movies_rdd.collect()
'''
['그린 북',
 '매트릭스',
 '토이 스토리',
 '캐스트 어웨이',
 '포드 V 페라리',
 '보헤미안 랩소디',
 '빽 투 더 퓨처',
 '반지의 제왕',
 '죽은 시인의 사회']
'''


# 리스트를 펼쳐 보는 네로우 트랜스포메이션
flat_movies = movies_rdd.flatMap(lambda x: x.split(" ")) # 어떻게 나눌 지 지정
flat_movies.collect()
'''
['그린',
 '북',
 '매트릭스',
 '토이',
 '스토리',
 '캐스트',
 '어웨이',
 '포드',
 'V',
 '페라리',
 '보헤미안',
 '랩소디',
 '빽',
 '투',
 '더',
 '퓨처',
 '반지의',
 '제왕',
 '죽은',
 '시인의',
 '사회']
'''


filtered_movies = flat_movies.filter(lambda x: x != "매트릭스") # 필터링
filtered_movies.collect()
'''
['그린',
 '북',
 '토이',
 '스토리',
 '캐스트',
 '어웨이',
 '포드',
 'V',
 '페라리',
 '보헤미안',
 '랩소디',
 '빽',
 '투',
 '더',
 '퓨처',
 '반지의',
 '제왕',
 '죽은',
 '시인의',
 '사회']
'''


# 집합과 관련된 트랜스포메이션
num1 = sc.parallelize([1, 2, 3, 4])
num2 = sc.parallelize([4, 5, 6, 7, 8, 9, 10])
num1.intersection(num2).collect()
'''
[4]
'''


num1.union(num2).collect() # 모든 값이 합쳐진다.
'''
[1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10]
'''


num1.subtract(num2).collect() # 중복된 데이터가 사라진다.
'''
[2, 1, 3]
'''


num_union = num1.union(num2)
num_union.sample(True, .5, seed=6).collect() # True: 리샘플링 여부 / .5: 샘플링 할 비율
'''
[2, 10]
'''


# 와이드 트랜스포메이션
foods = sc.parallelize(["짜장면", "마라탕", "짬뽕", "떡볶이", "쌀국수", "짬뽕", "짜장면", "짜장면", "짜장면",  "라면", "우동", "라면", "치킨", "돈까스", "회", "햄버거", "피자"])
foods_group = foods.groupBy(lambda x: x[0]) # 그룹핑
res = foods_group.collect()
for (k, v) in res:
    print(k, list(v))
'''
짜 ['짜장면', '짜장면', '짜장면', '짜장면']
마 ['마라탕']
짬 ['짬뽕', '짬뽕']
떡 ['떡볶이']
쌀 ['쌀국수']
라 ['라면', '라면']
우 ['우동']
치 ['치킨']
돈 ['돈까스']
회 ['회']
햄 ['햄버거']
피 ['피자']
'''


nums = sc.parallelize([1,2,3,4,5,6,7,8,9,10])
list(nums.groupBy(lambda x: x % 2).collect()[1][1])
'''
[2, 4, 6, 8, 10]
'''
```

# 액션(Actions)
- 결과값을 연산하여 리스트나 파이썬 오브젝트를 출력하거나 저장한다.
- 즉시 실행(Eager Execution)이 된다는 특징이 있다.

## 함수
- `collect()`
- `count()`
- `countByValue()`
- `take()`
- `top()`
- `reduce()`
- `fold()`
- `foreach()`

## 코드

```py
from pyspark import SparkConf, SparkContext


# 스파크 컨텍스트 생성
conf = SparkConf().setMaster("local").setAppName("transformations_actions")
sc = SparkContext(conf = conf)
sc.getConf().getAll() # 설정 리스트 확인
'''
[('spark.master', 'local'),
 ('spark.app.id', 'local-1639210625743'),
 ('spark.rdd.compress', 'True'),
 ('spark.serializer.objectStreamReset', '100'),
 ('spark.driver.port', '64503'),
 ('spark.submit.pyFiles', ''),
 ('spark.executor.id', 'driver'),
 ('spark.app.name', 'transformations_actions'),
 ('spark.submit.deployMode', 'client'),
 ('spark.ui.showConsoleProgress', 'true'),
 ('spark.driver.host', '192.168.200.112'),
 ('spark.app.startTime', '1639210624537')]
'''


# sc.stop() # 스파크 컨텍스트 중단
# sc = SparkContext(conf=conf) # 다시 생성하여야 사용 가능

foods = sc.parallelize(["짜장면", "마라탕", "짬뽕", "떡볶이", "쌀국수", "짬뽕", "짜장면", "짜장면", "짜장면",  "라면", "우동", "라면"]) # 텍스트파일 함수와 같이 RDD를 생성
foods
'''
ParallelCollectionRDD[0] at readRDDFromFile at PythonRDD.scala:274
'''


# RDD 데이터 확인
# 벨류를 모두 가져올 수 있는데 디버깅이나 개발 환경에서만 사용해야하고 실제 프로덕트 상황에서 지양해야한다.
# 데이터를 모두 가져오기 때문에 너무 낭비가 되기 때문에 스파크를 쓰는 의미가 없어진다.
foods.collect()
'''
['짜장면', '마라탕', '짬뽕', '떡볶이', '쌀국수', '짬뽕', '짜장면', '짜장면', '짜장면', '라면', '우동', '라면']
'''


foods.countByValue() # 벨류 값 계산하여 키 벨류 형태로 전시
'''
defaultdict(int,
            {'짜장면': 4,
             '마라탕': 1,
             '짬뽕': 2,
             '떡볶이': 1,
             '쌀국수': 1,
             '라면': 2,
             '우동': 1})
'''


foods.take(3) # 3개의 엘리먼트만 전시
'''
['짜장면', '마라탕', '짬뽕']
'''


foods.first() # 첫 값 전시
'''
'짜장면'
'''


foods.count() # 엘리먼트 갯수 전시
'''
12
'''


foods.distinct().count() # 엘리먼트 갯수 유니크하여 전시
'''
7
'''


# foreach: 요소를 하나하나 꺼내어 하나의 함수를 작용시키는데 쓰인다.
foods.foreach(lambda x: print(x))
# 실행시켜도 리턴값은 없지만, 워커 노드에서 실행이 되기 때문에 현재 드라이버 프로그램에서는 보이지 않는다.
# RDD를 연산하거나 로그를 세이브할 때 유용하게 쓰인다.
```