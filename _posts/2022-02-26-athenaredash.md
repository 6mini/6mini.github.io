---
title: '[아테나] 아마존 아테나(Amazon Athena) + 리대시(Redash) 연동'
description: S3에 적재된 데이터를 핸들링하기 위한 아마존 아테나와 리대시 연동하는 방법
categories:
 - Data Engineering
tags: [데이터 엔지니어링, 아테나, 리대시]
mathjax: enable
---

# 개요

![스크린샷 2022-02-24 오후 1 53 55](https://user-images.githubusercontent.com/79494088/155460475-111c14f3-7420-4ecd-bf41-a0153fac2339.png){: width="80%"}

- S3에 데이터 웨어하우스 형태로 데일리 적재를 시작했다.

![image](https://user-images.githubusercontent.com/79494088/155460817-acc7e2db-6e8e-42bb-9f90-1ce2391dd436.png)

- 아테나(Athena)를 이용하여 파티션 구분도 확인했고, 테이블도 만들 수 있는 상태이다.
- 회사에서 사용하는 분석 및 시각화 툴인 리대시(Redash)를 이용하기 위해서 아테나와 연동하는 작업을 진행할 것이다.

# IAM

## 정책 연결
- 아테나로 쿼리를 실행하고 데이터가 포함되어있는 S3 버킷에 액세스할 수 있는 권한이 있는 IAM 사용자를 생성한다.

![스크린샷 2022-02-24 오후 4 44 21](https://user-images.githubusercontent.com/79494088/155480629-25600d5b-eb68-48c5-b3b0-fca546c2e1dc.png){: width="80%"}

- `AWSQuicksightAthenaAccess` 정책을 연결한다.

# 리대시(Redash)

## 데이터 소스 생성

![image](https://user-images.githubusercontent.com/79494088/155480980-7fc23848-63c1-45e1-8b2f-4dfce57897b9.png)

- 설정에서 새로운 데이터 소스를 추가한다.

![image](https://user-images.githubusercontent.com/79494088/155483446-53416968-fcc1-4e50-be2c-e4f412a2e8f6.png){: width="80%"}

- 필드값을 채운다.
    - Name: 데이터 소스명
    - AWS Region: 필자의 경우 서울 `ap-northeast-2`
    - AWS Access Key, AWS Secret Key: IAM 키
    - S3 Staging (Query Results) Bucket Path: 아테나 쿼리 결과 위치

# 결과

![스크린샷 2022-02-24 오후 6 13 11](https://user-images.githubusercontent.com/79494088/155494445-40ef8da6-b5fb-4e0e-91a8-7fa5023e35f7.png){: width="80%"}

- 아테나에서 만들었던 테이블이 전시되며 쿼리문을 통해 접근이 가능하다.
- 이제 즐분!