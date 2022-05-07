---
title: '[SQL] 집계 함수(COUNT, SUM, AVG, MIN/MAX) 및 GROUP BY, ORDER BY'
description: '[백문이불여일타] 데이터 분석을 위한 중급 SQL 중 집계함수 간략 정리 및 해커랭크 문제 풀이'
categories:
 - Data Engineering
tags: [SQL, 데이터 엔지니어링]
---

# 간략 정리

Id|Name|Visits
1|A|1
2|A|2
3|B|3
4|C|5
5|NULL|NULL

## COUNT

```sql
SELECT COUNT(*)
FROM sample;
> 5

SELECT COUNT(Name)
FROM sample;
> 4

SELECT COUNT(DISTINCT Name)
FROM sample;
> 3
```

## SUM

```sql
SELECT SUM(Visits)
FROM sample;
```

## AVG

```sql
SELECT AVG(Visits)
FROM sample;
> (1+2+3+5) / 4 = 2.75

SELECT SUM(Visits)/COUNT(*) FROM sample;
> (1+2+3+5) / 5 = 2.2
```

## MAX/MIN
```sql
SELECT MAX(Visits)
FROM sample;

SELECT MIN(Visits)
FROM sample;
```

## GROUP BY

```sql
SELECT CategoryID, SUM(Price)
FROM Products
GROUP BY CategoryID;
```

(그룹화의 기준이 되는 컬럼은 SELECT 구문에 반드시 적어주기)

## HAVING

```sql
SELECT CategoryID, COUNT(*)
FROM Products
GROUP BY CategoryID
HAVING COUNT(*) <= 10
```

## ORDER BY

- 오름차순(Default)
    - ASC ascending
- 내림차순
    - DESC descending

```sql
SELECT *
FROM Products
WHERE Price >= 20
ORDER BY price DESC;
(Price가 20이상인 값들 중 비싼 순으로 정렬)
```

## LIMIT

```sql
SELECT *
FROM Products
ORDER BY price DESC
LIMIT 1;
```
(가장 비싼 물건 1개 출력)

# 해커랭크 문제 풀이

## Average Population
- Query the average population for all cities in CITY, rounded down to the nearest integer.
- Input Format
- The CITY table is described as follows:

<img width="369" alt="image" src="https://user-images.githubusercontent.com/79494088/162686774-597f5bea-4d3f-45e6-ac90-0370e5b8497b.png">

### 풀이 

```sql
SELECT ROUND(AVG(population))
FROM city
```

## Revising Aggregations - The Sum Function
- Query the total population of all cities in CITY where District is California.

### 풀이

```sql
select sum(population)
from city
where district = 'California'
```

## Revising Aggregations - Averages
- Query the average population of all cities in CITY where District is California.

### 풀이

```sql
select avg(population)
from city
where district = 'California'
```

## Revising Aggregations - The Count Function
- Query a count of the number of cities in CITY having a Population larger than 100,000.

### 풀이

```sql
select count(*)
from city
where population > 100000
```

## Population Density Difference
- Query the difference between the maximum and minimum populations in CITY.

### 풀이

```sql
select max(population) - min(population)
from city
```

## Weather Observation Station 4
- Find the difference between the total number of CITY entries in the table and the number of distinct CITY entries in the table.
- The STATION table is described as follows:

<img width="307" alt="image" src="https://user-images.githubusercontent.com/79494088/162911835-735c4d2a-59ec-4737-a783-14ebdc099c5e.png">

### 풀이

```sql
select count(city) - count(distinct city)
from station
```

## Top Earners
- [문제 바로가기](https://www.hackerrank.com/challenges/earnings-of-employees/problem?h_r=internal-search)
- We define an employee's total earnings to be their monthly salary * months worked, and the maximum total earnings to be the maximum total earnings for any employee in the Employee table. Write a query to find the maximum total earnings for all employees as well as the total number of employees who have maximum total earnings. Then print these values as 2 space-separated integers.
    - salary와 months를 곱하여 earnings를 생성한다.
    - group by를 통해 최고 금액과 갯수를 구한다.

<img width="352" alt="image" src="https://user-images.githubusercontent.com/79494088/163077932-fa4d8db0-8160-471d-b023-8167587a0b12.png">

### 풀이

```sql
select months * salary as earnings
     , count(*)
from employee
group by earnings
order by earnings desc
limit 1
```