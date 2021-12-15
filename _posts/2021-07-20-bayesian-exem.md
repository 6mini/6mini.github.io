---
title: '[통계] 예제로 이해하는 베이지안(Bayesian)'
description: 어려운 베이지안(Bayesian) 통계를 여러가지 문제 예제를 통해 이해하는 과정
categories:
 - Statistics
tags: [베이지안, 통계]
mathjax: enable
---

# 답을 알고 맞췄을 확률
> 객관식 문제를 푸는 과정에서, 학생은 답을 이미 알고 있거나 찍어야 한다.<br>
학생이 답을 알 확률은  𝑝 이고, 답을 찍어서 맞출 확률이  $1\over{m}$  이라고 할 때  𝑚 과  𝑝 를 입력받아 학생이 문제의 답을 알고 맞췄을 확률을 계산하는 함수를 작성하라.

## 추론과정
- P(알$\vert$정) : 정답인데, 답을 알고 맞췄을 확률
- P(알) = $p$
- P(몰) = 1 - $p$
- P(정$\vert$알) : 알아서 정답일 확률 = 1
- P(정$\vert$몰) : 모르는데 찍어서 맞출 확률 = $1\over{m}$
- P(알$\vert$정) = ${P(정\vert알) P(알)} \over {P(정\vert알) P(알) + P(정\vert몰) P(몰)}$

```python
def correct(p, m):
    # (1) * (p) / (1) * (p) + (1 / m) * (1 - p)
    # p / p + (1 / m) * (1 - p)
    ans = (p * m) / (p * (m - 1) + 1)
    return ans
    
correct(0.25, 4)
'''
0.5714285714285714
'''
```

**답 : 57%**

# 양성판정 시, 실제로 질병을 가지고 있을 확률

> 특정 질병을 가지고 있는 경우 99%의 확률로 탐지 할 수 있는 실험 방법이 있다.<br>
그러나 동시에 이 방법은, 1%의 확률로 질병이 없지만 질병이 있다고 진단 하는 경우도 있다.<br>
실제로 모든 인구중 0.5%만이 이 질병을 가지고 있다.<br>
특정 사람이 이 방법을 통해 질병이 있다고 진단 받았을 때, 실제로 질병을 가지고 있을 확률을 구하는 함수를 작성하라.

## 추론과정

- P(tpr$\vert$+) : 질병이 있다고 진단 받았을때, 실제로 질병을 가지고 있을 확률
- P(tpr) : 발병률 = prior = 0.005
- P(fpr) : 질병없음 = 1 - prior = 0.995
- P(+$\vert$tpr) : 걸렸는데 양성 = tpr = 0.99
- P(+$\vert$fpr) : 안걸렸는데 양성 = 1 - tpr = fpr = 0.01 
- p(tpr$\vert$+) = ${p(+\vert tpr) p(tpr)} \over {p(+\vert tpr) p(tpr) + p(+\vert fpr)p(fpr)}$ 

```python
def disease(prior, tpr, fpr):
    ans = (tpr * prior) / ((tpr * prior) + (fpr * (1-prior)))
    return ans
    
disease(0.005, 0.99, 0.001)
'''
0.832632464255677
'''
```

**답 : 83%**

# 왼손잡이일 때, 유죄일 확률
> At a certain stage of a criminal investigation, the inspector in charge is 60% convinced of the guild of a certain suspect.<br>
Suppose now that a new piece of evidence that shows that the criminal has a left-handedness is uncovered.<br>
If 20% of population possesses this characteristic, how certain of the guilt of the suspect should the inspector now be if it turns out that the suspect is among this group?<br>
범죄 수사의 특정 단계에서 담당 수사관은 특정 피의자의 유죄를 60% 확신합니다.<br>
이제 범인이 왼손잡이를 가지고 있다는 것을 보여주는 새로운 증거가 발견되었다고 가정해 보겠습니다.<br>
인구의 20%가 이런 특성을 가지고 있다면, 만약 용의자가 이 집단 안에 있다는 것이 밝혀진다면, 지금 조사관의 유죄는 얼마나 확신해야 할까요?

## 추론과정(1)
- P(유죄$\vert$왼손) : 왼손잡이 일 때, 유죄일 확률
- P(유죄) = 0.6
- P(무죄) = 0.4
- P(왼손$\vert$유죄) : 유죄인데 왼손일 확률 = 1
- P(왼손$\vert$무죄) : 무죄인데 왼손일 확률 = 0.2
- P(유죄$\vert$왼손) = ${P(왼손\vert유죄)P(유죄)} \over {P(왼손\vert유죄)P(유죄) + P(왼손\vert무죄)P(무죄)}$

```python
ans1 = 0.6 / (0.6 + (0.2 * 0.4))
ans1
'''
0.8823529411764707
'''
```

**답 : 88%**

> After that, the new evidence is subject to different possible interpretations, and in fact only shows that it is 90% likely that the criminal possess this characteristic.<br>
In this case how likely would it be that the suspect is guilty?<br>
그 후, 새로운 증거는 다른 해석의 대상이 되며, 사실 범인이 이러한 특성을 가지고 있을 가능성이 90%에 불과하다는 것을 보여줍니다.<br>
이 경우 용의자가 유죄일 확률이 얼마나 됩니까?

## 추론과정(2)
- P(유죄$\vert$새증거) : 새로운 증거의 특징을 가질 가능성이 90퍼센트 일 때, 유죄일 확률
- P(유죄) = ans1
- P(무죄) = 1 - ans1
- P(새증거$\vert$유죄) : 유죄이면서 새증거의 특징을 가졌을 확률 = 0.9
- P(새증거$\vert$무죄) : 무죄이면서 새증거의 특징을 가졌을 확률 = 0.1
- P(유죄$\vert$새증거) = ${P(새증거\vert유죄)P(유죄)} \over {P(새증거\vert유죄)P(유죄) + P(새증거/무죄)P(무죄)}$

```python
ans2 = 0.9 * ans1 / ((0.9 * ans1) + (0.1 * (1 - ans1)))
ans2
'''
0.9854014598540146
'''
```

**답 : 98%**