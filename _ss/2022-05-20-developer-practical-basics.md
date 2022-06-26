---
title: '[개발자 실무 기본기] 개요 및 환경 구축'
description: "인프런 '모든 개발자의 실무를 위한 필수 기본기 클래스' 강의 소개 및 실습 환경 구축"
categories:
 - Computer Science
tags: []
---

# 개요

## 강의 소개

### 실무에서 필요한 기본기의 핵심 영역
- 클린코드에서 바로 적용할만한 내용을 코드와 함께 학습한다.
- 객체 지향 프로그래밍의 기본적인 이해와 실습을 진행한다.
- 테스트 코드 작성법을 익히고 실습 프로젝트에 테스트 코드를 적용한다.
- 대표적인 아키텍처 패턴을 알아보고 클린 아키텍처를 직접 적용한다.
- 실무에서 사용할 깃(Git) 명령어를 실습과 함께 다룬다.
- 개발 지식을 학습한다.

## 실습 환경 구축
- 강의에서는 파이썬(python) 버전 3.10과 파이참(Pycharm)을 다루지만, 필자는 3.9 버전과 VSCODE로 진행할 것이다.
- 파이썬 기본 문법은 패스할 것이다.

# 강의 목차

## 소개 및 이용 가이드
- 강의 소개
- 강의 활용 가이드

## 실습 환경 구축하기
- 사전 준비
- 사전 파이썬 공부하기
- 파이썬 기본훑기

## 협업의 필수 Git, 실전에서 자주 사용되는 명령어
- Git 기초 돌아보기
- 작업 공간
- 브랜치
- 상황별 Git 다루는 법
- [log & reflog] 이전 commit 내역들과 변경사항을 확인하고 싶어요
- [restore & reset] 변경사항, 커밋을 초기화하고 싶어요
- [stash] 변경 사항을 커밋하기 보단 임시저장하고 싶어요
- [revert] 이전 커밋의 변경사항을 되돌리고 싶어요.
- [amend commit & rebase] 이전에 쌓인 커밋들을 변경하고 싶어요.
- [squash & rebase merge] 브랜치를 머지할 때 머지 커밋을 남기기 싫어요
- [cherry-pick] 다른 브랜치에 있는 커밋을 내 브랜치로 가져오고 싶어요.
- 실전 충돌(Conflict) 다루기 - 1,2
- 전략적으로 Git 사용하기 - Gitflow

## 깔끔한 코드를 위하여! 클린코드
- 클린 코드 - 네이밍
- 클린 코드 - 주석, 포맷팅
- 클린 코드 - 함수
- 클린 코드 - 클래스
- 클린 코드 - 에러 핸들링
- 클린 코드 - 코드 indent 줄이기(Guard Clausing, Polymorphism)

## 코드로 알아보는 객체 지향 프로그래밍
- 프로그래밍 패러다임 흐름 훑고가기 - 절차지향
- 프로그래밍 패러다임 흐름 훑고가기 - 객체 지향
- 프로그래밍 패러다임 흐름 훑고가기 - 함수형 프로그래밍
- 객체 지향의 기본적인 개념들 짚고 가기
- 코드로 알아보는 객체 지향의 특성
- 의존성 응집도 결합도

## (실습) 리팩토링을 통해 객체 지향 알아보기
- 리팩토링을 통해 객체 지향 알아보기

## 객체 지향 설계를 위한 SOLID 원칙
- SOLID - Single Responsibility
- SOLID - Open Closed
- SOLID - Liskov Substitution
- SOLID - Interface Segregation
- SOLID - Dependency Inversion

## 견고한 서비스를 위한 테스트 코드 작성하기
- 테스트 기본 이해하기
- 종류별 테스트 작성하기
- 의존성을 대체하는 테스트 더블
- TDD 기본 개념 익히기

## (실습) 프로젝트에 테스트 적용하기
- 프로젝트에 Test 적용하기

## 더 나은 설계를 위해, 소프트웨어 아키텍처 기초와 패턴 이해하기
- 아키텍처를 시작하기 전에
- 대표적인 소프트웨어 아키텍처 패턴 알아보기 - 레이어드 아키텍처
- 대표적인 소프트웨어 아키텍처 패턴 알아보기 - 헥사고날 아키텍처
- 대표적인 소프트웨어 아키텍처 패턴 알아보기 - 클린 아키텍처
- 코드로 알아보는 클린 아키텍처
- 모놀리스와 마이크로서비스 아키텍처

## (실습) 프로젝트에 아키텍처 적용하기
- 프로젝트에 아키텍처와 테스트 적용하기
- 프로젝트 더 디벨롭하기

## 회사에서 공부하면 좋을 개발 지식들
- 1.효율적으로 프로그램 운영하기
    - 프로세스와 스레드 기본
    - 병렬성과 동시성
    - 멀티 스레드와 멀티 프로세스
    - 동기와 비동기, 블락과 논블락
- 2.쉽고 빠르게 프로그램 배포하기
    - 가상화 기술과 도커
    - 배포와 CI/CD
- 3.서비스의 핵심 요소, 회원가입과 로그인 이해하기
    - 쿠키와 세션
    - 사용자 인증(Authentication)
- 4.추가
    - OSI 7계층과 TCP/IP 4계층 모델
    - 우리 회사는 어떻게 웹 서비스를 운영할까?
