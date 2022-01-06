---
title: '[통계] 가설 검정 방법(T-검정, 카이 제곱 검정)'
description: T-검정(T-test)의 조건과 그 외의 다른 가설 검정 방법, Type of Error의 구분, 카이 제곱 검정의 목적과 사용 예시, 모수통계와 비모수통계의 차이
categories:
 - Mathematics
tags: [T-검정, 카이 제곱 검정, 통계, 비모수 통계, Type of Error]
mathjax: enable
---

# T-검정(T-test)
- 이전 포스팅에서 알아보았던 T-검정은 그룹의 평균값에 대해서 비교하는 가설 검정 방법이었다.
- **T-검정이란 모집단의 분산이나 표준편차를 알지 못할 때, 표본으로부터 추정된 분산이나 표준 편차를 이용하여 두 모집단의 평균 차이를 알아보는 검정 방법**이다.
- 집단의 수는 최대 2개 까지 비교가 가능하며 3개 이상인 경우 분산 분석(ANOVA)를 사용한다.
- 하지만 T-검정을 사용하기 위해서는 몇 가지 조건이 가정되어야 한다.


## T-검정의 조건

![image](https://user-images.githubusercontent.com/79494088/145703109-baffb780-e16d-4b53-8ca0-f414708f0a92.png)

### 1. 독립성
- 독립 변수의 그룹군은 서로 독립적이어야 한다.
	- 대응 표본일 경우 대응 표본 T-검정을 실행한다.

### 2. 등분산성
- 독립 변수에 따른 종속 변수 분포의 분산은 유사한 값을 가진다.
	- 분산이 서로 다를 경우 자유도를 수정한 독립 표본 T-검정을 실행한다.

#### 등분산성 테스트
- [scipy.stats.normaltest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html#scipy-stats-normaltest)

```python
sample2 = np.random.normal(size = 1000) # normal 분포
normaltest(sample2)
```

![](https://images.velog.io/images/6mini/post/21c64580-e823-4c20-ad2e-721a7291e8ff/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-07-16%2015.43.14.png)

### 3. 정규성
- 독립 변수에 따른 종속 변수는 정규 분포를 만족해야한다.
	- 정규 분포가 아닐 경우 Mann-Whitney test를 실행한다.

# Type of Error

![스크린샷 2021-08-09 18 19 22](https://user-images.githubusercontent.com/79494088/128684730-18653a54-fe1c-4cf0-a045-ee7977f15bd9.png)

- 술을 안마셨는데 음주탐지기 반응이 나왔을 경우: 제 1종 오류(FP)
- 술을 마셨는데 음주탐지기 미반응이 나왔을 경우: 제 2종 오류(FN)
	- 참고로 음주 탐지기의 경우 제 2종 오류를 줄이는 것이 중요하다. => Recall을 우선해야 한다.

{% include ad.html %}

# 비모수 통계(Non-Parametric Methods)
- 수집된 자료가 정규 분포하지 않은 경우에 사용된다.
	- 모집단의 확률 분포가 정규 분포를 따르지 않는 경우
	- 표본 수가 작아 모집단의 정규 분포를 가정하기 어려운 경우
	- 측정한 자료의 수준이 명목형인 경우
- 모집단이 특정 확률 분포를 따른다는 전제를 하지 않는 방식이다.
- 모수 추정치(Parameter estimation)가 필요하지 않기 때문에 비모수(Non-Parametric)라고 부른다.
	- 예
		- 연속성이 없는 데이터(Categorical Data)
		- 극단적 outlier가 있는 데이터

## 비모수적 평균 비교법(Kruskal-Wallis Test)

```python
# Kruskal-Wallis H-test: 2개 이상 그룹의 중위 랭크를 통한 차이 비교(extended X2)
# 샘플 수가 > 5 일 때 좋다.
from scipy.stats import kruskal

x1 = [1, 3, 4, 8, 9]
y1 = [1, 4, 6, 7, 7]
kruskal(x1, y1) # 약간은 다르지만, 유의한 차이는 아니다.
```

![](https://images.velog.io/images/6mini/post/3a233382-ee70-40db-a815-40f84b5b74af/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-07-16%2015.57.52.png)

```python
x2 = [12, 15, 18]
y2 = [24, 25, 26]
z = [40, 40]  # 3번째 그룹은 사이즈가 다름
kruskal(x2, y2, z)
```

![](https://images.velog.io/images/6mini/post/87ae30e8-4151-4b4f-b3fc-5a73c586bad3/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-07-16%2015.58.08.png)

# $\chi^2$ 검정(Test)

![image](https://user-images.githubusercontent.com/79494088/145704201-80e0240c-f9db-420b-90a4-1b66bda86928.png)

- 카이 제곱 분포에 기초한 통계적 방법으로, 관찰된 빈도가 기대되는 빈도와 의미있게 다른 지의 여부를 검정하기 위해 사용되는 검정 방법이다.

## 1 샘플 $\chi^2$ 검정

![](https://images.velog.io/images/6mini/post/beaf6070-f2ab-4aed-acbb-c2250f1bd2ae/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-07-16%2016.03.25.png)

 - 주어진 데이터가 특정 예상되는 분포와 동일한 분포를 나타내는지에 대한 가설 검정이다.

## $\chi^2$ 통계치의 계산식

$$\chi^2$ = $\sum$ $\frac{(observed_i-expected_i)^2}{(expected_i)}$$

- 각 차이의 값을 제곱하는 것으로, 모든 값을 양수로 만들고 관측과 예측값의 차이를 더 강조하는 효과가 있다.
- [카이 제곱 테스트 방법 포스팅 바로가기](https://6mini.github.io/did%20unknown/2021/07/16/didunk3/)

![image](https://user-images.githubusercontent.com/79494088/145704143-0f6be7da-d53e-45bf-b620-0cde424993b0.png)

# 자유도(Degrees of Freedom)

$$1 sample = value -1$$

$$2 sample = (row - 1) * (column - 1)$$

- 얼마나 다양한 축으로 움직일 수 있는 지를 말하며, 주어진 조건 안에서 통계적인 추정을 할 때 표본이 되는 자료 중 모집잔에 대해 정보를 주는 독립적인 자료의 수를 말한다.


- [이론 및 T검정](http://www.incodom.kr/이론_및_T검정)
- [T-검정(T-test)](https://velog.io/@rsj9987/T-검정)
- [우리는 이미 일상 생활에 머신러닝의 개념을 적용하고 있었다…](https://www.andrewahn.co/product/using-ml-concepts-in-real-life/)
- [2 모수검정vs비모수검정](https://m.blog.naver.com/nlboman/23354659)
- [자유도(degree of freedom)](https://www.scienceall.com/자유도degree-of-freedom-2/)