---
title: '[Deep Learning] 어텐션 메커니즘(Attention Mechanism)'
description: 신경망의 성능을 높이기 위한 메커니즘이자, 이제는 AI 분야에서 대세 모듈로서 사용되고 있는 트랜스포머의 기반이 되는 어텐션 메커니즘
categories:
 - Deep Learning
tags: [Deep Learning]
mathjax: enable
---

# Attention
- seq2seq 모델은 인코더에서 입력 시퀀스를 컨텍스트 벡터라는 하나의 고정된 크기의 벡터 표현으로 압축되고, 디코더는 이 컨텍스트 벡터를 통해서 출력 시퀀스를 만들었다.
- 이러한 RNN에 기반한 seq2seq 모델에는 문제가 있다.
  - 고정된 크기의 벡터에 모든 정보를 압축하려고 하니까 정보 손실이 발생한다.
  - RNN의 고질적인 기울기 소실(Vanishing Gradient) 문제가 존재한다.
- 이 때문에 기계 번역 분야에서 문장이 길면 번역 품질이 떨어지는 현상으로 나타났다.
- 이를 위한 대안으로 입력 시퀀스가 길어지면 출력 시퀀스의 정확도가 떨어지는 것을 보정해주기 위해 등장한 기법인 Attention이 있다.

## Attention Idea
- 디코더에서 출력 단어를 예측하는 매 시점(time step) 마다, 인코더에서의 전체 입력 문장을 다시 한번 참고한다는 점이다.
- 전체 입력 문장을 전부 다 동일한 비율로 참고하는 것이 아니라, 해당 시점에서 예측해야 할 단어와 연관이 있는 입력 단어 부분을 좀 더 집중해서 보게 된다.

## Attention Function

![image](https://user-images.githubusercontent.com/79494088/139058213-3edc94b0-ce57-4875-a4b8-0522fa6fca0b.png)

- 어텐션 함수: Attention(Q, K, V) = Attention Value
- 주어진 Query에 대해 모든 Key와의 유사도를 각각 구한다.
- 구해낸 이 유사도를 키와 mapping되어 있는 각각의 '값(Value)'에 반영한다.
- 유사도가 반영된 값을 모두 더해서 리턴한다.
- 이를 어텐션 값이라고 한다.

## Dot-Product Attention
- 수식적으로 이해하기 쉬운 닷-프로덕트 어텐션을 통해 이해해보자.

![image](https://user-images.githubusercontent.com/79494088/139059048-61df9c93-997c-4e89-91f6-699d70c7e27c.png)

- 위 그림은 디코더의 세번째 LSTM 셀에서 출력 단어를 예측할 때, 어텐션 메커니즘을 사용하는 모습이다.
- 소프트맥스 함수를 통해 나온 결과값은 I, am, a, student 단어 각각이 출력 단어를 예측할 때 얼마나 도움이 되는지의 정도를 수치화한 값이다.
- 각 입력 단어가 디코더의 예측에 도움이 되는 정도를 수치화하여 측정되면 이를 하나의 정보로 담아서 디코더로 전송된다.

## Attention Mechanism

### Attention Score를 구한다.

![image](https://user-images.githubusercontent.com/79494088/139059569-8461faf9-e243-4a8d-acf6-ec545a2e5730.png)

- 인코더의 time step을 각각 1, 2, ... N 이라고 했을 때, 인코더의 hidden state를 각각 $h_{1}, h_{2}, ... h_N$ 이라고 한다.
- 디코더의 time step $t$에서 hidden state를 $S_t$라고 한다.
- 인코더의 은닉 상태와 디코더의 은닉 상태의 차원이 같다고 가정한다.
- 시점 $t$에서 출력 단어를 예측하기 위해 디코더의 셀은 이전 시점인 $t-1$, $t-2$에 나온 출력 단어가 필요하다.
- 어텐션 메커니즘에서는 출력 단어 예측에 '어텐션 값'이라는 새로운 값이 필요하다.
- $t$번째 단어를 예측하기 위한 어텐션 값을 $a_t$라고 정의한다.
- 지금부터 배우는 모든 여정은 $a_t$를 구하기 위한 여정이다.
- 그 여정의 첫걸음은 어텐션 스코어를 구하는 일이다.
- 어텐션 스코어란 현재 디코더의 시점 $t$에서 단어를 예측하기 위해, 인코더의 모든 은닉 상태 각각이 디코더의 현 시점의 은닉 상태 $s_t$와 얼마나 유사한지 판단하는 스코어 값이다.
- 닷-프로덕트 어텐션에서는 이 스코어 값을 구하기 위해 $s_t$를 transpose(전치)하고 각 은닉 상태와 dot product(내적)fmf tngodgksek.
- 모든 어텐션 스코어 값은 스칼라이다.

- s_t와 인코더 i번째 은닉 상태의 어텐션 스코어 계산방법

![image](https://user-images.githubusercontent.com/79494088/139060889-4d50df0c-99e0-45cb-81ad-8c1b9c6cc46f.png)

- 어텐션 스코어 함수 정의

$$ score(s_{t},\ h_{i}) = s_{t}^Th_{i} $$

- $s_t$와 인코더의 모든 은닉 상태의 어텐션 스코어의 모음값을 $e^t$라고 정의하면 $e^t$의 수식은,

$$ e^{t}=[s_{t}^Th_{1},...,s_{t}^Th_{N}] $$

### Softmax fuc을 통해 Attention Distribution을 구한다.

![image](https://user-images.githubusercontent.com/79494088/139061273-1eb0b6ce-7e6f-4714-9c4d-c33b9a101804.png)

- e^t에 소프트맥스 함수를 적용하여, 모든 값을 합하면 1이 되는 확률 분포를 얻는다.
- 이를 Attention Distribution이라고 하며, 각각의 값은 Attention Weight라고 한다.
- 디코더의 시점 $t$에서의 어텐션 가중치의 모음값인 어텐션 부포를 $α^t$라고 할 때, $α^t$를 식으로 정의하면,

$$ α^{t} = softmax(e^{t}) $$

### 각 인코더의 어텐션 가중치와 은닉 상태를 가중합하여 Attention Value를 구한다.

![image](https://user-images.githubusercontent.com/79494088/139061683-be8befc1-9c3a-4754-aad0-c09884525dc5.png)

- 지금까지 준비해온 정보를 하나로 합치는 단계이다.
- 어텐션의 최종 결과값을 얻기 위해 각 인코더의 은닉 상태와 어텐션 가중치값을 곱하고, 최종적으로 모두 더한다.
- 요약하면 가중합을 한다고 말할 수 있다.
- 어텐션의 최정 결과. 즉, 어텐션 함수의 출력값인 어텐션 값(Attention Value) $a_t$에 대한 식은,

$$ a_{t}=\sum_{i=1}^{N} α_{i}^{t}h_{i} $$

- 이러한 어텐션 값은 종종 인코더의 문맥을 포함하고 있다고 하여, Context Vector라고도 불린다.

### 어텐션 값과 디코더의 t시점의 은닉 상태를 Concatenate한다.

![image](https://user-images.githubusercontent.com/79494088/139062109-b164916d-fd29-4a89-a640-ee9bbc52342f.png)

- 어텐션 값이 구해지면 어텐션 메커니즘은 $a_t$를 $s_t$와 결합(concatenate)하여 하나의 벡터로 만드는 작업을 수행한다.
- 이를 $v_{t}$라고 정의하고 이 것을 $\hat{y}$ 예측 연산의 입력으로 사용하므로서 인코더로부터 얻은 정보를 활용하여 $\hat{y}$를 좀 더 잘 예측할 수 있게 된다.
- 이것이 어텐션 메커니즘의 핵심이다.

### 출력층 연산의 입력이 되는 $\tilde{s}_{t}$를 계산한다.

![image](https://user-images.githubusercontent.com/79494088/139062598-fc9c8de8-c99b-4eb4-b0c3-67c7092a8519.png)

- 가중치 행렬과 곱한 후에 하이퍼볼릭탄젠트 함수를 지나도록하여 출력층 연산을 위한 새로운 벡터인 $\tilde{s}_{t}$를 얻는다.
- 이를 식으로 표현하면, (식에서 $W_{c}$는 학습 가능한 가중치 행렬)

$$ \tilde{s}_{t} = \tanh(\mathbf{W_{c}}[{a}_t;{s}_t] + b_{c}) $$

### $\tilde{s}_{t}$를 출력층의 입력으로 사용한다.

$$ \widehat{y}_t = \text{Softmax}\left( W_y\tilde{s}_t + b_y \right) $$

## 다양한 종류의 어텐션

![image](https://user-images.githubusercontent.com/79494088/139195586-98310c1a-7690-4b5f-b465-14584dd363c1.png)

# Reference
- [어텐션을 소개한 논문](https://arxiv.org/pdf/1409.0473.pdf)
- [추천하는 참고 자료](http://docs.likejazz.com/attention/)
- [추천하는 참고 자료2](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2015/slides/lec14.neubig.seq_to_seq.pdf)
- [추천하는 참고 자료3](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
- [추천하는 참고 자료4](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)