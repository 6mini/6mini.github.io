---
title: '[빅데이터 엔지니어링] 프로젝트(1) 개요'
description: Apache Spark를 정복하기 위한 빅데이터 엔지니어링 프로젝트의 시작, 데이터 엔지니어링과 그 역사 및 트렌드, ETL과 ELT, 배치 및 스트림 프로세싱, 데이터 플로우 오케스트레이션, 간략한 프로젝트 계획
categories:
 - Project
tags: [데이터 엔지니어링 프로젝트]
---

# 데이터 엔지니어링이란?

## 데이터 엔지니어링의 목적
- 데이터 기반 의사결정을 위한 인프라를 구축하는 것이다.

### 비즈니스 의사결정
- 가격책정
- 모니터링
- 분석

### 서비스 운영/개선
- A/B 테스트
- UI/UX
- 운영/자동화

## 데이터 엔지니어링의 전망
- 기업에서는 분석보다 데이터 엔지니어링을 더 필요로 한다.
- 데이터를 이용해서 인사이트를 추출하는 업무의 대부분이 데이터 엔지니어링이다.
- 복잡한 데이터 모델을 만드는 것 보다, 좋은 데이터를 모으고 잘 관리하는 것이 훨씬 효율적으로 성과를 내는 방법이라할 수 있다.
- 데이터는 앞으로 계속 증가할 것이고, 데이터 엔지니어링은 더욱 중요해질 것이다.

# 모던 데이터 엔지니어링 아키텍쳐

## 과거
- 컴퓨팅 파워와 용량이 비쌌다.
- 데이터의 용도가 정해져 있었다.(앱이면 앱, 분석이면 분석)
- 데이터가 나올 곳도 정해져 있었다.

### 데이터 관리 방식
- 데이터의 형식(스키마)을 미리 만들어야 했다.
- 데이터의 변동이 별로 없었다.
- 효율적인 데이터베이스 모델링이 중요했다.

#### ETL
- Extract: 추출
- Transform: 스키마에 맞게 변환
- Load: 데이터베이스에 적재

### 다양해지는 데이터 형식
- 점점 데이터로 할 수 있는 일이 다양해지고 형태를 예측하기 불가능해지면서 스키마를 정의하기 힘들어졌다.
    - 실시간성을 요구하는 기능
    - 빨라지는 기능의 추가
    - 실시간 로그
    - 비정형 데이터(텍스트, 오디오, 비디오)
    - 서드 파티 데이터(데이터가 한군데가 아닌 여러군데에서 나온다.)

### 저렴해지는 컴퓨팅 파워
- 이젠 최대한 많은 데이터를 미리 저장해두고 많은 양의 프로세싱을 더할 수 있게 되었다.
- 일반적인 회사에선 이제 컴퓨팅 파워에 대한 비용 최적화보다, 비즈니스와 속도를 최적화하는 쪽의 이득이 더 크다.

## 현재

### 데이터를 운용하는 방식

![image](https://user-images.githubusercontent.com/79494088/143839973-86357358-89d6-478c-bbb7-79fa910c3672.png)

- 기존의 ETL 방식에서 ELT 방식의 아키텍쳐로 점점 변환하고 있다.

#### 예

![image](https://user-images.githubusercontent.com/79494088/143840124-791edc9d-c67d-4393-a45f-d51e7849f5bb.png)

- 시스템의 복잡도에 따라 데이터 추출과 적재를 한번에 하기도 한다.

### 데이터 인프라 트렌드
- 클라우드 웨어하우스로 옮겨가는 추세이다.
    - Snowflake, Google Big Query와 같은 솔루션
- Hadoop에서 Databricks, Presto같은 다음 세대로 넘어가는 추세이다.
- 실시간 빅데이터 처리에 대한 수요가 늘고있다.
    - Steream Processing
- ETL 방식에서 ELT 방식으로 변환하고 있다.
- 데이터 파이프라인 자체가 복잡해지고 의존성도 관리하기 힘들어지다 보니, Dataflow를 자동화하는 추세이다.
    - Airflow
- 데이터 분석 팀을 두기보다, 누구나 분석할 수 있도록 데이터 툴이 진화하고 있다.
- 데이터 플랫폼이 점점 중앙화되고 있다.
    - access control, data book

### 모던 데이터 아키텍쳐 해부

#### 데이터 아키텍쳐 분야를 크게 6가지로 나누어본다면,

![image](https://user-images.githubusercontent.com/79494088/143841221-d7a87ec7-4123-45a2-9ba4-2496e2e02a1e.png)

#### 데이터가 흘러가는 과정

![image](https://user-images.githubusercontent.com/79494088/143841493-6e5b4e20-4f1f-4024-9e01-d511de7599a3.png)

#### 데이터 엔지니어링 도구들

![image](https://user-images.githubusercontent.com/79494088/143841561-e06671cf-e205-4e03-8114-a0a6c30ea52e.png)

- 특징이 수집 및 변환 분야와 처리 분야에 집중되어 있는 것을 볼 수 있다.
- 소스나 저장, 쿼리같은 경우에는 서비스 레벨보다는 로우레벨의 문제들을 푸는 분야이다.
    - 소스나 쿼리 분야에서는 서비스 관점에서 풀 엔지니어링 문제가 거의 없다.
    - 저장분야에서는 로우레벨의 문제, 즉 데이터를 어떻게 효율적으로 저장할 것인지에 대한 문제이다.
- 일반적인 엔지니어링은 수집 및 변환, 데이터 처리에 집중이 되어있다.
- 이번에 다룰 툴은 Airflow, Kafka, Spark, Flink, SparkML 등이 있다.
    - Spark: 데이터 병렬-분산 처리
    - Airflow: 데이터 오케스트레이션
    - Kafka: 이벤트 스트리밍
    - Flink: 분산 스트림 프로세싱

# Batch & Stream Processing

## 배치 프로세싱
- 배치(Batch): 일괄
- 배치 프로세싱(Batch Processing): 일괄 처리
- 즉, 많은 양의 데이터를 정해진 시간에 한꺼번에 처리하는 것이다.
    - 한정된 대량의 데이터가 필요하다.
    - 특정 시간에 따라 처리를 한다.
    - 일괄 처리를 한다는 특징이 있다.
- 이는 전통적으로 쓰이는 데이터 처리 방법이다.

### 언제 사용할까
- 실시간성을 보장하지 않아도 될 때
- 데이터를 한거번에 처리할 수 있을 때
- ML학습같이 무거운 처리를 할 때

#### 예
- 매일 다음 14일의 수요와 공급을 예측
- 매주 사이트에서 관심을 보인 유저들에게 마케팅 이메일 전송
- 매주 발행하는 뉴스레터
- 매주 새로운 데이터로 머신러닝 알고리즘 학습
- 매일 아침 웹 스크래핑/크롤링
- 매달 월급 지급
- 지금까지 필자가 했던 모든 프로젝트가 배치 프로세싱이다.

## 스트림 프로세싱

![image](https://user-images.githubusercontent.com/79494088/143843938-121ba855-b086-4b8c-8507-6c8a96f128b0.png)

- 실시간으로 쏟아지는 데이터를 계속 처리하는 것이다.
- 이벤트가 생길 때마다, 데이터가 들어올 때마다 처리한다.

### 불규칙적인 데이터가 들어오는 환경을 가정

![image](https://user-images.githubusercontent.com/79494088/143844163-ed2ce172-ac88-4e59-9031-b9c10cd76a3b.png)

- 위 그림과 같이 여러개의 이벤트가 한꺼번에 들어오거나, 오랜 시간동안 이벤트가 하나도 들어오지 않을 경우가 있다.

![image](https://user-images.githubusercontent.com/79494088/143844330-d2dc7f08-3694-46a2-935f-c1ba194dee9f.png)

- 이 때, 배치 프로세싱을 진행하면 배치당 처리하는 데이터 수가 달라지면서 리소스를 비효율적으로 사용하게 된다.

![image](https://user-images.githubusercontent.com/79494088/143844453-b3f7e756-7919-41d5-b6f1-5c4e3ac6548e.png)

- 하지만 스트림 프로세싱으로 데이터가 생성되어 요청이 들어올 때마다 처리할 수 있다.

### 언제 사용할까
- 실시간성을 보장해야 될 때
- 데이터가 여러 소스로부터 들어올 때
- 데이터가 가끔 들어오거나 지속적으로 들어올 때
- Rule-based와 같은 가벼운 처리를 할 때

#### 예
- 사기 거래 탐지
- 이상 탐지
- 실시간 알림
- 비즈니스 모니터링
- 실시간 수요/공급 측정 및 가격책정
- 실시간 기능이 들어가는 어플리케이션

## 배치 vs 스트림

![image](https://user-images.githubusercontent.com/79494088/143844954-866c98ad-a624-414f-b47d-ab5257062df6.png)

- 일반적인 배치 처리 플로우는 데이터를 모아, 데이터베이스에서 읽어서 처리한 후 다시 데이터베이스에 담는다.

![image](https://user-images.githubusercontent.com/79494088/143845091-e1dd53af-361d-48e2-ae33-32aa7edb2e46.png)

- 일반적인 스트림 처리 플로우는 데이터가 들어올 때(ingest)마다 쿼리/처리 후 State를 업데이트 한 후 데이터베이스에 담는다.

## 마이크로 배치

![image](https://user-images.githubusercontent.com/79494088/143845512-cc6b3bee-d18c-4de5-9e47-0b32c456ca53.png)

- 데이터를 조금씩 모아 프로세싱하는 방식이며 배치 프로세싱을 잘게 쪼개서 스트리밍을 흉내내는 방식이다.

# Dataflow Orchestration

## 오케스트레이션이란
- 오케스트라처럼 데이터 테스크를 지휘하는 느낌이다.
    - 테스크를 스케줄링한다.
    - 분산 실행을 돕는다.
    - 테스크 간 의존성을 관리한다.

## 필요성
- 요즘 트렌드를 보면 오케스트레이션의 필요성을 알 수 있다.
    - 서비스가 커지면서 데이터 플랫폼의 복잡도가 커진다.
    - 데이터가 사용자와 직접 연관되는 경우가 늘어난다.<br>(워크플로우가 망가지면 서비스도 망가진다.)
    - 테스크 하나하나가 중요해졌다.
    - 테스크 간의 의존성이 생겼다.

### 오케스트레이션 없이 문제가 생겼을 때

<img width="945" alt="image" src="https://user-images.githubusercontent.com/79494088/143881900-3021a46e-416d-4964-911b-cf6ff35ba179.png">

- 중간 Task2 에서 오류가 발생 시 시간이 지체된다.

### 오케스트레이션이 있었다면
- Task2 에러발생 시 로그 남기고 알림이 울린다.
- 실패 시나리오에 따라 Task 2가 다시 실행되고 성공한다.
- Task 4가 실행된다.

### 복잡한 의존성
- 실 서비스에선 데이터 테스크가 더 복잡하게 얽히게 된다.

<img width="808" alt="image" src="https://user-images.githubusercontent.com/79494088/143882361-34bf0673-1dac-41c3-bc7f-f57acfca84e6.png">

<img width="1586" alt="image" src="https://user-images.githubusercontent.com/79494088/143882439-a7fdc99e-c20d-455f-ba74-a0555242ca5c.png">

## Apach Airflow

<img width="586" alt="image" src="https://user-images.githubusercontent.com/79494088/143882530-85e18646-372d-4efc-abd2-47026511d3cb.png">

- 오케스트레이션을 도와주는 대표적인 툴이 바로 Apache Airflow이다.

<img width="789" alt="image" src="https://user-images.githubusercontent.com/79494088/143882675-c8f57263-a336-46be-893e-600c84b6918c.png">

- 데이터 테스크를 관리하는 대시보드가 지원되어, 실시간으로 워크플로우를 확인할 수 있다.

# 프로젝트 소개

## 데이터 기반 의사결정 사례: Uber

<img width="573" alt="image" src="https://user-images.githubusercontent.com/79494088/143883472-54c48d8c-bf43-4de7-8a54-d90d7fba3d69.png">

- 과거 데이터와 실시간 데이터를 기반으로 수요와 공급 예측을 통해 자동으로 가격을 조정해 고객이 기다리는 시간(ETA)를 최소화했다.

### 실 서비스 활용 예

<img width="846" alt="image" src="https://user-images.githubusercontent.com/79494088/143883615-3070e56d-c608-4001-87de-6263a47ba957.png">

- 예측을 위해 쉽게 머신러닝 알고리즘을 학습하고 배포할수 있는 플랫폼이 만들어졌다.

<img width="728" alt="image" src="https://user-images.githubusercontent.com/79494088/143884205-c27ea52b-c01b-4268-8037-fb57dcaab832.png">

- 우버 내부에서 배달에 걸리는 시간 등을 예측하는데 사용되었다.

<img width="1086" alt="image" src="https://user-images.githubusercontent.com/79494088/143884295-793ca7d2-3a32-4981-a849-cef1de548f2c.png">

<img width="1085" alt="image" src="https://user-images.githubusercontent.com/79494088/143884870-83126ec7-f230-4b5a-8a77-646af2e99a47.png">

- 머신러닝 학습같이 무거운 프로세스는 오프라인 배치 프로세싱으로 학습하고, 학습된 데이터로 배달에 걸리는 시간을 실시간으로 예측하는데 사용됐다.

## 프로젝트 목표
- 배치 파이프라인과 스트림 파이프라인을 동시에 사용하는 ML 데이터 학습 + 서빙 파이프라인을 만들 것이다.

### 배치 파이프라인

<img width="1013" alt="image" src="https://user-images.githubusercontent.com/79494088/143886114-b3eeae7b-0088-4dcc-a467-b5d44d629245.png">

- 데이터 프리프로세싱과 ML학습은 주기적으로 실시할 것이기 때문에 Aiflow Orchestration을 이용하여 관리할 것이다.

### 스트림 파이프라인

<img width="1017" alt="image" src="https://user-images.githubusercontent.com/79494088/143886608-3abe8e85-b5f5-4ebc-a835-221da352a7f5.png">

### 배치 + 스트림 파이프라인

<img width="1267" alt="image" src="https://user-images.githubusercontent.com/79494088/143887116-804c03f0-5d75-4987-809b-a0dd3a2f1b75.png">

# 참조
- [Emerging Architectures for Modern Data Infrastructure](https://a16z.com/2020/10/15/the-emerging-architectures-for-modern-data-infrastructure/)
- [What is Apache Flink? — Architecture](https://flink.apache.org/flink-architecture.html)
- [Meet Michelangelo: Uber’s Machine Learning Platform](https://eng.uber.com/michelangelo-machine-learning-platform/)
- [Review notes of ML PLatforms — Uber Michelangelo](https://medium.com/@nlauchande/review-notes-of-ml-platforms-uber-michelangelo-e133eb6031da)
- [Apache Airflow](https://airflow.apache.org/)