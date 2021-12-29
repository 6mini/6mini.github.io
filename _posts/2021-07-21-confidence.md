---
title: '[통계] 예제로 이해하는 신뢰구간'
description: 어려운 신뢰구간을 예제 문제를 통해 이해하는 과정
categories:
 - Mathematics
tags: [통계, 신뢰구간]
---

- [신뢰구간 포스팅 바로가기](https://6mini.github.io/statistics/2021/07/19/ci/)

# 데이터셋

```python
df.head()
```

![](https://images.velog.io/images/6mini/post/5502586c-ce73-4d72-a3a1-1f625544f89f/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-07-21%2015.14.06.png)

# 샘플 생성

```python
s1 = df.sample(n = 20, 
               random_state = 42)
s2 = df.sample(n = 200, 
               random_state = 42)
print(s1.head())
print(s2.head())
```

![](https://images.velog.io/images/6mini/post/fe0927ec-3005-4748-8888-86c23b441611/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-07-21%2015.15.38.png)

![](https://images.velog.io/images/6mini/post/5d463d99-d7bb-42dd-852a-3446a0502980/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-07-21%2015.16.06.png)

# 95% 신뢰구간

```python
from scipy.stats import t
import numpy as np

# 표본의 크기
n1 = len(s1['오존(ppm)'])
n2 = len(s2['오존(ppm)'])

# 자유도
dof1 = n1 - 1
dof2 = n2 - 1

# 표본의 평균
mean1 = np.mean(s1['오존(ppm)']) 
mean2 = np.mean(s2['오존(ppm)'])

# 표본의 표준편차
sample_std1 = np.std(s1['오존(ppm)'], ddof = 1)
sample_std2 = np.std(s2['오존(ppm)'], ddof = 1)

# 표준 오차
std_err1 = sample_std1 / n1 ** 0.5
std_err2 = sample_std2 / n2 ** 0.5

CI1 = t.interval(.95, dof1, loc = mean1, scale = std_err1) 
CI2 = t.interval(.95, dof2, loc = mean2, scale = std_err2)

print("s1의 95% 신뢰구간: ", CI1)
print("s2의 95% 신뢰구간: ", CI2)
```

![](https://images.velog.io/images/6mini/post/d58d62da-08ae-4ee0-89cf-838143d3b417/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-07-21%2015.20.30.png)

# 시각화

```python
import matplotlib.pyplot as plt
%matplotlib inline

pop_mean = df['오존(ppm)'].mean() # population : 모집단

x = ['s1', 's2']
y = [mean1, mean2]
err = [std_err1, std_err2] # 표준오차

plt.bar(x, y, yerr = err, capsize = 7, color = ['dodgerblue', 'orange'], width = 0.8);
plt.axhline(pop_mean, 0, 1, color='#4000c7', linestyle='--', linewidth='1');
plt.axhline(mean1, 0.15, 0.34, color='black', linestyle='-', linewidth='2');
plt.axhline(mean2, 0.65, 0.86, color='black', linestyle='-', linewidth='2');
```

![](https://images.velog.io/images/6mini/post/7f9ff5cc-28c4-465c-840f-df51b60a771e/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-07-21%2015.23.55.png)