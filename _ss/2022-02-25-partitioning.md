---
title: '[아테나] 아테나(Athena)를 위한 S3 데이터 파티셔닝'
description: 
categories:
 - Data Engineering
tags: [데이터 엔지니어링, 아테나, S3]
mathjax: enable
---

# 하이브(Hive)란?
- 하이브는 하둡 에코 시스템 중 데이터를 모델링하고 프로세싱하는 경우, 가장 많이 사용하는 데이터 웨어하우징용 솔루션이다.
- RDB의 데이터베이스, 테이블과 같은 형태로 HDFS에 저장된 데이터의 구조를 정의하는 방법을 제공하며, 이 데이터를 대상으로 SQL과 유사한 HiveQL 쿼리를 이용하여 데이터를 조회하는 방법을 제공한다.
- 가장 큰 특징으로 메타스토어(Metastore)라는 것이 존재하는데, 하이브는 기존 RDB와 다르게 미리 스키마를 정의하고 그 틀에 맞게 데이터를 입력하는 것이 아닌, 데이터를 저장하고 거기에 스키마를 입하는(메타스토어에 입력하는) 것이 가장 큰 특징이다.

## 하이브의 구성 요소

![image](https://user-images.githubusercontent.com/79494088/155632104-25602f0f-c6c8-4bdd-8664-a454d39d9416.png){: width="80%"}

- UI
    - 사용자가 쿼리 및 기타 작업을 시스템에 제출하는 사용자 인터페이스
    - CLI, Beeline, JDBC 등
- Driver
    - 쿼리를 입력받고 작업을 처리
    - 사용자 세션을 구현하고, JDBC/ODBC 인터페이스 API 제공
- Compiler
    - 메타스토어를 참고하여 쿼리 구문을 분석하고 실행계획을 생성
- Metastore
    - DB, 테이블, 파티션의 정보를 저장
- Execution Engine
    - 컴파일러에 의해 생성된 실행 계획을 실행

### 하이브의 실행 순서
1. 사용자가 제출한 쿼리문을 드라이버가 컴파일러에 요청하여 메타스토어의 정보를 이용해 처리에 적합한 형태로 컴파일한다.
2. 컴파일된 SQL을 실행 엔진으로 실행한다.
3. 리소스 매니저가 클러스터의 자원을 적절히 활용하여 실행한다.
4. 실행 중 사용하는 원천 데이터는 HDFS 등의 저장 장치를 이용한다.
5. 실행 결과를 사용자에게 반환한다.

## 하이브의 등장 배경
- 하이브는 SQL을 하둡에서 사용하기 위한 프로젝트로 시작됐다.
- 하둡의 MR을 JAVA로 표현하기보다 익숙한 SQL로 데이터를 핸들링하는 것이 편하기에 나오게 된 개념이다.

## 하이브 메타스토어(Metastore)란?
- 데이터 파일의 물리적인 위치, 하이브의 테이블, 파티션과 관련된 메타 정보를 모두 저장하는 곳이다.
- 하이브의 메타스토어는 빅데이터의 '우선 데이터가 있고, 나중에 스키마를 입힌다.'의 개념에 딱 맞는 개념이다.
- 하이브는 기존 RDBMS와 달리 데이터를 삽입 후 스키마를 입히게 되는데, 그 때 스키마 정보를 메타스토어에서 참조한다.
- HDFS에 있는 데이터에 스키마를 참조할 수 있는 데이터베이스이며, 기존의 RDBMS로 메타스토어를 지정한다.

# 아테나(Athena)에서 데이터 파티셔닝(Partitioning)
- 데이터를 분할하면 각 쿼리가 스캔하는 데이터의 양을 제한하여 성능을 향상시키고 비용을 절감할 수 있다.
- 키를 기준으로 데이터를 분할할 수 있으나, 일반적으로 시간을 기준으로 데이터가 분할되어 다중 레벨 파티셔닝 체계가 형성되는 경우가 많다.
    - 매 시간 데이터를 수집할 경우 연, 월, 일, 시를 기준으로 분할할 수 있다.
- 아테나는 하이브 스타일 파티션을 사용할 수 있다.
    - `country=us/...` 또는 `year=2021/month=01/day=26/...`
    - 경로에 파티션 키의 이름과 각 경로가 나타내는 값이 포함되어야 한다.
- 파티션을 나눈 테이블에서 새 하이브 파티션을 로드하려면 `MSCK REPAIR TABLE` 명령을 사용한다.
- 필자의 경우 아래와 같이 파티션을 구성했다.
    - `large-category/medium-category/year={year}/month={month}/day={day}/hour={hour}/name.parquet`

## 파티셔닝 데이터로 테이블 생성 및 로드

### 테이블 생성

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS table_name (
    date timestamp,
    rank int,
    code string,
    name string,
    full_price int,
    current_price int,
    star DOUBLE,
    reviews int
)PARTITIONED BY (year string, month string, day string, hour string)
STORED AS PARQUET LOCATION 's3://bucket-name/large-category/medium-category' tblproperties("parquet.compress"="SNAPPY")
```

### 데이터 로드

```sql
MSCK REPAIR TABLE table_name
```

### 데이터 쿼리

