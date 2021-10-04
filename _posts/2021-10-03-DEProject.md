---
title: '[DE Project] Data Pipeline 구축 1. Spotify API'
description: 스포티파이 API 이용 데이터 파이프라인 구축 토이 프로젝트 Architecture 및 과정
categories:
 - Project
tags: [Project, Data Engineering, Data Pipeline, Spotify API, Architecture, 데이터 파이프라인, 데이터 엔지니어링, 스포티파이, 아키텍쳐]
---

# Data Engineering의 필요성

## 문제해결을 위한 가설 검증 단계

![스크린샷 2021-10-02 22 46 33](https://user-images.githubusercontent.com/79494088/135719088-d1463a01-e0d2-4506-9591-aed1489eb8fa.png)

- 모든 business가 동일한 Data 분석 환경을 갖출 수 없으며 성장 단계에 따라 선택 집중해야 하는 분석 환경이 다르다.
- 유저 경험이 중요한 business의 경우 처음부터 Data 시스템 구축이 성공의 열쇠

![스크린샷 2021-10-02 22 52 44](https://user-images.githubusercontent.com/79494088/135719318-c9928830-d51e-4d0f-a5a3-6c09a466cf2e.png)

- 이커머스는 마케팅, CRM, 물류 Data 분석을 통해 전략 수집
- 처음부터 모든 인력을 갖출 필요는 없고 성장 단계별로 필요한 분석 환경을 갖추는 것이 키

![스크린샷 2021-10-03 00 03 59](https://user-images.githubusercontent.com/79494088/135722202-c55ab782-c320-450a-933e-0eb12d6b2b45.png)

# Data Architecture 시 고려사항

## business 모델 상 가장 중요한 Data
- **비용 대비 business impact가 가장 높은 Data**

## Data Governance

### 원칙(Principle)
- Data를 유지 관리하기 위한 가이드
- 보안, 품질, 변경관리

### 조직(Organization)
- Data를 관리할 조직의 역할과 책임
- Data 관리자, Data Architect

### Process
- Data 관리를 위한 시스템
- 작업 절차, 모니터 및 측정

## 유연하고 변화 가능한 환경 구축
- 특정 기술 및 솔루션에 얽매여져 있지 않고 새로운 테크를 빠르게 적용할 수 있는 Architecture를 만드는 것
- 생성되는 Data의 형식이 변화할 수 있는 것처럼 그에 맞는 tool과 solution도 빠르게 변화할 수 있는 시스템을 구축하는 것

## 실시간 Data handling이 가능한 시스템
- 밀리세컨 단위의 스트리밍 Data가 됐건 하루에 한 번 업데이트 되는 Data든 Data Architecture는 모든 스피드의 Data를 handling
  - Real Time Streaming Data Processing
  - Cronjob
  - Serverless Triggered Data Processing

## Security
- 위험 요소를 파악하여 Data를 안전하게 보관

## 셀프 서비스 환경 구축
- 한명만 Access할 수 있는 시스템은 확장성이 없는 Data
  - BI Tools
  - Query System for Analysis
  - Front-end data APP

# Data 시스템의 옵션

## API
- 다양한 플랫폼 및 소프트웨어는 API를 통해 Data를 주고 받을 수 있는 환경을 구축하여 생태계를 생성

## RDB
- 관계형 Data베이스

## NoSQL

## Hadoop / Spark / Presto
- Distributed Storage System / MapReduce를 통한 병렬 처리
- Spark
  - Hadoop의 진화버전
  - Real Time Data를 Processing하기에 최적화
  - Python을 통한 API를 제공하여 APP 생성
  - SQL Query 환경 서포트

## Serverless Framework
- Triggered by http requests, db events, queuing services
- Pay as you use
- Form of functions
- 3rd party APP 및 다양한 API를 통해 Data를 수집 정제하는데 유용

# Data Pipeline
- Data를 한 장소에서 다른 장소로 옮긴다.
  - API -> DB
  - DB -> DB
  - DB -> BI

## 필요한 경우
- 다양한 Data 소스로부터 많은 Data를 생성하고 저장하는 서비스
- data silos : 마케팅, 어카운팅, 세일즈, 오퍼레이션 등 각 영역의 Data가 서로 고립
- 실시간 혹은 높은 수준의 Data 분석이 필요한 business 모델
- 클라우드 환경으로 Data 저장

<img width="1536" alt="스크린샷 2021-10-03 00 23 34" src="https://user-images.githubusercontent.com/79494088/135722781-0d6403c5-a831-4f52-8366-565ec34dc40e.png">

## 구축시 고려사항
- Scalability : Data가 기하급수적으로 늘어났을 때도 작동하는가?
- Stability : 에러, Data 플로우 등 다양한 Monitering 관리
- Security : Data 이동 간 보안에 대한 리스크는 무엇인가?

# 자동화
- Data를 추출, 수집, 정제하는 Processing을 최소한의 사람 인풋으로 머신이 운영하는 것

## 고려사항

### Data Processing Step

<img width="948" alt="스크린샷 2021-10-03 00 32 59" src="https://user-images.githubusercontent.com/79494088/135723050-701839fe-df85-4832-a9af-fd3ff693deca.png">

### 에러 handling 및 Monitering

<img width="843" alt="스크린샷 2021-10-03 00 33 58" src="https://user-images.githubusercontent.com/79494088/135723082-adb8171d-8693-409e-bff4-c8fcac39575b.png">


- Trigger / Scheduling

<img width="947" alt="스크린샷 2021-10-03 00 35 05" src="https://user-images.githubusercontent.com/79494088/135723122-6a0f8a97-6d3e-40b4-a429-8275977bdef2.png">

# End to End Architecture

## Data Lake

<img width="1189" alt="스크린샷 2021-10-03 00 36 37" src="https://user-images.githubusercontent.com/79494088/135723155-c40905d3-8f13-45e5-a822-6a397d5a9a70.png">

## 예시

### Netflix

<img width="1098" alt="스크린샷 2021-10-03 00 39 28" src="https://user-images.githubusercontent.com/79494088/135723265-1a7e0257-0117-4635-8be8-60e316f517b5.png">


### Uber

<img width="1065" alt="스크린샷 2021-10-03 00 40 02" src="https://user-images.githubusercontent.com/79494088/135723282-ec7f793f-4222-476a-80e6-87ec3bb4e3ab.png">

# Spotify project data Architecture

## Ad hoc VS Automated
- Ad hoc 분석 환경 구축은 서비스를 지속적으로 빠르게 변화시키기 위해 필수적인 요소
- 이니셜 Data 삽입, Data Backfill 등을 위해 Ad hoc Data Processing 시스템 구축 필요
- Automated : 이벤트, 스케쥴 등 Trigger를 통해 자동화 시스템 구축

## Artist 관련 Data 수집 Process

<img width="1158" alt="스크린샷 2021-10-03 00 48 19" src="https://user-images.githubusercontent.com/79494088/135723563-359e3b77-112f-45dc-8ce2-b0212f344cb8.png">

## Data 분석 환경 구축

<img width="934" alt="스크린샷 2021-10-03 00 56 09" src="https://user-images.githubusercontent.com/79494088/135723803-fa0fb15a-6659-45bd-a7cd-06d58511856c.png">

## 서비스 관련 Data Process

<img width="1029" alt="스크린샷 2021-10-03 00 57 33" src="https://user-images.githubusercontent.com/79494088/135723839-4aac7d2f-4b28-48f9-a1d5-fd5d48f9634c.png">


# API

## Authentication VS Authorization
- Authentication : 정체가 맞다는 증명
- Authorization : 어떠한 액션을 허용

## OAuth 2.0

<img width="988" alt="스크린샷 2021-10-03 01 58 30" src="https://user-images.githubusercontent.com/79494088/135725707-f06e4489-5446-40b9-a8a3-6cff1cbcedff.png">


## Spotify

```py
import sys
import requests
import base64
import json
import logging

client_id = ""
client_secret = ""


def main():

    headers = get_headers(client_id, client_secret)

    ## Spotify Search API
    params = {
        "q": "BTS",
        "type": "artist",
        "limit": "5"
    }

    r = requests.get("https://api.spotify.com/v1/search", params=params, headers=headers)


    try:
        r = requests.get("https://api.spotify.com/v1/search", params=params, headers=headers)

    except:
        logging.error(r.text)
        sys.exit(1)


    r = requests.get("https://api.spotify.com/v1/search", params=params, headers=headers)

    if r.status_code != 200:
        logging.error(r.text)

        if r.status_code == 429:

            retry_after = json.loads(r.headers)['Retry-After']
            time.sleep(int(retry_after))

            r = requests.get("https://api.spotify.com/v1/search", params=params, headers=headers)

        ## access_token expired
        elif r.status_code == 401:

            headers = get_headers(client_id, client_secret)
            r = requests.get("https://api.spotify.com/v1/search", params=params, headers=headers)

        else:
            sys.exit(1)


    # Get BTS' Albums

    r = requests.get("https://api.spotify.com/v1/artists/3Nrfpe0tUJi4K4DXYWgMUX/albums", headers=headers)

    raw = json.loads(r.text)

    total = raw['total']
    offset = raw['offset']
    limit = raw['limit']
    next = raw['next']

    albums = []
    albums.extend(raw['items'])

    ## 난 100개만 뽑아 오겠다
    while next: 

        r = requests.get(raw['next'], headers=headers)
        raw = json.loads(r.text)
        next = raw['next']
        print(next)

        albums.extend(raw['items'])
        count = len(albums)

    print(len(albums))

def get_headers(client_id, client_secret):

    endpoint = "https://accounts.spotify.com/api/token"
    encoded = base64.b64encode("{}:{}".format(client_id, client_secret).encode('utf-8')).decode('ascii')

    headers = {
        "Authorization": "Basic {}".format(encoded)
    }

    payload = {
        "grant_type": "client_credentials"
    }

    r = requests.post(endpoint, data=payload, headers=headers)

    access_token = json.loads(r.text)['access_token']

    headers = {
        "Authorization": "Bearer {}".format(access_token)
    }

    return headers

if __name__=='__main__':
    main()

```