---
title: '[고급 SQL] 윈도우 함수(Window Functions)란?'
description: "[데이터 분석을 위한 고급 SQL] "
categories:
 - SQL
tags: [SQL, 윈도우 함수, Window Functions]
---

# 윈도우 함수(Window Functions)란?
- 여러 개의 행을 하나의 집합으로 묶은 것을 하나의 윈도우라고 한다.

```sql
함수(컬럼) OVER (PARTITION BY 컬럼 ORDER BY 컬럼)
```

- 함수를 쓰고 그 뒤에 over을 쓴 후 괄호 안에는 `partition by` 혹은 `order by` 구문을 사용(둘 다 사용하거나 둘 중 하나만 사용할 수 있음)한다.
- `partition by`를 쓸 경우, 뒤에 쓰인 컬럼을 기준으로 데이터가 분할된다.
- `order by`를 쓸 경우, 뒤에 쓰인 컬럼을 기준으로 데이터가 정렬된다.
- 윈도우 함수는 `over (partition by ~ order by ~ )`구문 앞에 집계함수를 사용할 수도 있고, 집계 함수가 아닌 함수를 사용할 수도 있다.

## 집계 함수
1. count() over (partition by ~ order by ~ )
2. sum() over (partition by ~ order by ~ )
3. min() over (partition by ~ order by ~ )
4. max() over (partition by ~ order by ~ )

## 비집계 함수
1. 순위를 정하는 함수(row_number, rank, dense_rank)
2. 데이터 위치를 바꾸는 함수(lag, lead)
    - row_number() over (partition by ~ order by ~ )
    - lead() over (partition by ~ order by ~ )

- 윈도우 함수는 사용할 수 있는 위치가 한정적이다.(SELECT 절 / ORDER BY 절에서만 사용가능)
- 조건을 걸 수 있는 WHERE절에서 사용할 수 없다면 윈도우 함수를 활용하여 연산을 한 결과에 어떻게 조건을 걸까? => 서브쿼리를 사용하여 조건을 걸 수 있다.

# 윈도우 함수 사용 예제

## 집계 함수

### MAX(column) OVER (PARTITION BY column)
- employee

Id|Name|Salary|DepartmentId
---|---|---|---
1|이순신|20000|1
2|김유신|30000|2
3|강감찬|40000|2
4|이지수|50000|1

```sql
SELECT Id
     , Name
     , Salary
     , DepartmentId
     , MAX(Salary) OVER (PARTITION BY DepartmentId) AS MaxSalary
FROM employee
```

Id|Name|Salary|DepartmentId|MaxSalary
---|---|---|---|---
1|이순신|20000|1|50000
2|김유신|30000|2|40000
3|강감찬|40000|2|40000
4|이지수|50000|1|50000

### SUM(column) OVER (PARTITION BY column)
- elevator

Id|Name|Kg|Line
---|---|---|---
1|이순신|70|1
1|김유신|80|2
2|강감찬|90|1
2|이지수|100|2

```sql
SELECT Id
     , Name
     , Kg
     , Line
     , SUM(Kg) OVER (PARTITION BY Id ORDER BY Line) AS CumSum
FROM elevator
```

Id|Name|Kg|Line|CumSum
---|---|---|---|---
1|이순신|70|1|70
1|김유신|80|2|150
2|강감찬|90|1|90
2|이지수|100|2|190

## 순위를 정하는 함수
- ROW_NUMBER(), RANK(), DENSE_RANK()

### 예시

val|name
---|---
1|이순신
1|김유신
2|강감찬
3|이지수
3|임꺽정
4|윤봉길

```sql
SELECT val
     , ROW_NUMBER() OVER (ORDER BY val) AS row_number
     , RANK() OVER (ORDER BY val) AS rank
     , DENSE_RANK() OVER (ORDER BY val) AS dense_rank
FROM Sample
```

val|row_number|rank|dense_rank
---|---|---|---
1|1|1|1
1|2|1|1
2|3|3|2
2|4|4|2
3|5|4|3
4|6|6|4

- `row_number`는 어떻게 해서든 123456으로 매겨준다.(중복이어도)
- `rank`는 1등과 4등을 동시에 줘서 2등과 5등없이 3등, 6등으로 스킵한다.
- `dense_rank`는 동시에 같은 등수를 주지만, 순서대로 순위를 매겨준다.

## 데이터 위치를 바꾸는 함수
- LAG(), LEAD()
    - LAG(): 데이터 앞에 있는 값을 가져온다.
    - LEAD(): 데이터 뒤에 있는 값을 가져온다.

```sql
LAG(컬럼) OVER (PARTITION BY 컬럼 ORDER BY 컬럼)
LAG(컬럼, 칸 수) OVER (PARTITION BY 컬럼 ORDER BY 컬럼)
LAG(컬럼, 칸 수, Default) OVER (PARTITION BY 컬럼 ORDER BY 컬럼)

LEAD(컬럼) OVER (PARTITION BY 컬럼 ORDER BY 컬럼)
LEAD(컬럼, 칸 수) OVER (PARTITION BY 컬럼 ORDER BY 컬럼)
LEAD(컬럼, 칸 수, Default) OVER (PARTITION BY 컬럼 ORDER BY 컬럼)
```

### 예시
- weather

Id|Date|Temperature
---|---|---
1|2018-01-01|10
1|2018-01-02|20
1|2018-01-03|30

```sql
SELECT Id
     , Date
     , Temperature
     , LAG(Temperature) OVER (PARTITION BY Id ORDER BY Date) AS LagTemp
     , LEAD(Temperature) OVER (PARTITION BY Id ORDER BY Date) AS LeadTemp
FROM weather
```

Id|Date|Temperature|LagTemp|LeadTemp
---|---|---|---|---
1|2018-01-01|10|NULL|20
2|2018-01-02|20|10|30
3|2018-01-03|30|20|NULL

# 리트코드 문제 풀이

## 180. Consecutive Numbers
- [바로가기](https://leetcode.com/problems/consecutive-numbers/)

```sql
WITH leaders AS(
                SELECT num
                     , LEAD(num, 1) OVER (ORDER BY id) AS lead1
                     , LEAD(num, 2) OVER (ORDER BY id) AS lead2
                FROM logs
)

SELECT DISTINCT num AS ConsecutiveNums 
FROM leaders
WHERE num = lead1 AND num = lead2
```

## 184. Department Highest Salary
- [바로가기](https://leetcode.com/problems/department-highest-salary/)

```sql
WITH tables AS(
                SELECT d.name AS Department
                     , e.name AS Employee
                     , e.Salary AS Salary
                     , RANK() OVER (PARTITION BY d.name ORDER BY e.Salary DESC) AS Rank
                FROM employee AS e
                    INNER JOIN department AS d ON e.departmentId = d.id
)

SELECT Department, Employee, Salary
FROM tables
WHERE rank = '1'
```

## 185. Department Top Three Salaries
- [바로가기](https://leetcode.com/problems/department-top-three-salaries/)

```sql
WITH tables AS(
                SELECT d.id AS id
                     , d.name AS Department
                     , e.name AS Employee
                     , e.Salary AS Salary
                     , DENSE_RANK() OVER (PARTITION BY d.name ORDER BY e.Salary DESC) AS Rank
                FROM employee AS e
                    INNER JOIN department AS d ON e.departmentId = d.id
)

SELECT Department, Employee, Salary
FROM tables
WHERE rank = '1' OR rank = '2' OR rank = '3'
```