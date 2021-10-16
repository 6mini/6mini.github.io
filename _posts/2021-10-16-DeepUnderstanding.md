---
title: '[Deep Learning] Understanding'
description: 인공지능에 대한 이해
categories:
 - Deep Learning
tags: [Deep Learning, Machine Learning, Data, Neural Network, History]
mathjax: enable
---

# Concept

## Introduction

![image](https://user-images.githubusercontent.com/79494088/137547821-1638eaaf-bd0e-4395-9b90-891bbbc2b10c.png)

## History of AI

![image](https://user-images.githubusercontent.com/79494088/137547943-50dbef5b-fa3b-4f0d-8145-d812bd378eb6.png)

## Machine VS Deep

![image](https://user-images.githubusercontent.com/79494088/137548212-8e19466f-c9be-4c8b-80f0-9f4c22bfc709.png)

![image](https://user-images.githubusercontent.com/79494088/137548234-cf115321-9906-4dc4-9bb8-cc1c44d90c99.png)

![image](https://user-images.githubusercontent.com/79494088/137548264-0389c55c-3867-48d9-becc-2700a0fc29b2.png)

## 종류

### Supervised Learning
- 입력 data와 정답을 이용한 학습
- Classification, Regression

### Unsupervised Learning
- 입력 data만을 이용한 학습
- Clustering, Compression

### Reinforcement Learning(강화)
- Trial and error
- Action selection, Policy learning

![image](https://user-images.githubusercontent.com/79494088/137548593-7d35b753-ee48-4ba5-9848-a1ff2f4a860c.png)

## Training & Testing

![image](https://user-images.githubusercontent.com/79494088/137548701-6f829c31-996d-4f71-b1b9-2752d0b6fdca.png)


# Data
- Data depend on the type of the problem to solve.

![image](https://user-images.githubusercontent.com/79494088/137549106-0d634085-7f4d-4a19-9f9f-129ca75d0cdf.png)

## Important
- data가 많아질 수록 성능은 계속 좋아진다.

![image](https://user-images.githubusercontent.com/79494088/137549247-645833c5-b22b-4078-a486-379db0c98a31.png)

## Active Learning

![스크린샷 2021-10-16 05 26 24](https://user-images.githubusercontent.com/79494088/137549368-b5261f93-22ed-44fb-b9ce-99fe9ce84359.png)

## Good Data VS Bad Data

### unbiased

![image](https://user-images.githubusercontent.com/79494088/137549470-dda4fdb9-e906-4856-884b-c1845872a2a9.png)

## Label perfect
- 질이 중요하다.

![image](https://user-images.githubusercontent.com/79494088/137549602-33e2d0c3-4edc-482e-a033-005479b1b923.png)

## Data-Centric AI

![image](https://user-images.githubusercontent.com/79494088/137549720-5b91fadd-0658-44b9-83f0-3cf75c71aedb.png)

![image](https://user-images.githubusercontent.com/79494088/137549769-28084d4c-0be4-4fba-81e1-a519553e7c7d.png)

- The following are about equally effective
  - Clean up the noise
  - Collect another many new examples
- With a data centric view, there is a significant of room for improvement in problems with < 10,000 examples!

![image](https://user-images.githubusercontent.com/79494088/137550026-6952edd1-ad0e-4fad-ad58-b86e2bbd5ce5.png)

# Artificial Neural Network

![image](https://user-images.githubusercontent.com/79494088/137550127-15b69a97-c143-4c8c-a1c3-3178bdf58166.png)

- input data: 주어진 숫자
- label: 나온 답
- weight: 네모와 세모
- weight 값을 기계가 스스로 학습을 통해 찾아내도록 하는 것이 neural network를 이용한 기계학습이 하는 일

## Perceptron
- $y = Wx + b$

![스크린샷 2021-10-16 05 37 39](https://user-images.githubusercontent.com/79494088/137550648-423d9eb2-c5a0-4e83-8a5f-5890555391a8.png)

- 이러한 weight 값을 기계 스스로 찾을 수 있도록 해주는 과정

### Logical XNOR
- 레이어를 많이 쌓으면 된다.

![image](https://user-images.githubusercontent.com/79494088/137551215-43ab8dca-f319-4947-942e-64f432fa5410.png)

![image](https://user-images.githubusercontent.com/79494088/137551284-01e4ac88-8ed1-4633-b4a8-efe44da3923b.png)

### SLP & MLP

![image](https://user-images.githubusercontent.com/79494088/137551402-a0d73400-c34e-416f-be6c-57f73aec3a99.png)

## Deep Learning
- Deep Neural Network을 이용한 Machine Learning 방법
- Hidden layer 수가 최소 2개 이상인 network

![image](https://user-images.githubusercontent.com/79494088/137551549-6193b74f-3e9b-4d34-b215-f6975562d7c3.png)

# Training Neural Networks
- 최적의 weight 값
  - 잘 모르겠으니 일단 아무 값이나 넣고 시작

![image](https://user-images.githubusercontent.com/79494088/137575979-8b7903b0-f678-4ee9-a19b-31ee39e54a2e.png)

- Neural Network이 얼마나 잘 맞추는지에 대한 척도가 필요함
  - Loss Func
  - Cost Func
- 많이 쓰는 방법: 차이의 제곱(MSE?)
- Loss Function의 값이 줄어들도록 weight 값을 조금씩 바꾸는 것
  - 미분

## 미분
- w = w0 에서의 미분 값 = 이 점에서의 접선의 기울기
  - w0에서 미분값이 -2라면, w를 w0에서 왼쪽으로 아주 조금 움직이면 그 2배만큼 L값이 증가
- Loss를 w로 미분하고, 미분값이 가리키는 방향의 반대방향으로 아주 조금씩 w를 바꿔나가면 Loss를 감소시킬 수 있다.

![image](https://user-images.githubusercontent.com/79494088/137576152-eb8215dd-5fb1-4d82-ae9e-8080c9d081f3.png)

### Gradient Descent
- 마치 산에 서 눈 가리고 내려가는 느낌
- Loss Func의 Gradient를 이용하여 weight을 update하는 방법

![image](https://user-images.githubusercontent.com/79494088/137576536-b4a98956-d736-4612-bc7b-2c9a09a01b44.png)

#### Back Propagation
- Loss로부터 거꾸로 한 단계식 미분 값을 구하고 이 값들을 chain rule에 의해 곱해가면서 weight에 대한 gradient를 구하는 방법

![image](https://user-images.githubusercontent.com/79494088/137576621-91854499-a68d-48c1-a90a-62c97aacb7e6.png)

## Key Components
- Data
- Model
- Loss
- Algorithm

# Historical Review

![image](https://user-images.githubusercontent.com/79494088/137576689-8f24f654-dcc2-428e-b185-646b8feb240c.png)

## 2012: AlexNet
- 이 일을 계기로 CNN이 많이 쓰이게 됨

![image](https://user-images.githubusercontent.com/79494088/137576713-3f5683c9-2e53-4b94-940b-25a83a51e251.png)

## 2013: Atari

![image](https://user-images.githubusercontent.com/79494088/137576743-a8d4928b-b50c-4c7b-a740-864d26a96a34.png)

## 2014

### Attention

![image](https://user-images.githubusercontent.com/79494088/137576776-19c6171a-c8bc-4968-bc1b-42116a5bcf9e.png)

### Adam Optimizer

![image](https://user-images.githubusercontent.com/79494088/137576806-b52d6d57-eef7-4cbd-a67b-2d82108781e0.png)

## 2015

### Generative Adversarial Networks(GANs)

![image](https://user-images.githubusercontent.com/79494088/137576838-b2e51673-04ab-4246-a01c-e0c99080b2a7.png)

### Residual Networks(ResNet)

![image](https://user-images.githubusercontent.com/79494088/137576859-c536c829-50a3-4ae8-85ec-5c91e8efce4e.png)

## 2016: AlphaGo

![image](https://user-images.githubusercontent.com/79494088/137576874-de2a35a1-e63a-4ede-b974-043f439fa297.png)

## 2017: Transformer
- Attention is All You Need

![image](https://user-images.githubusercontent.com/79494088/137576889-d03c3649-4055-425e-b0dd-a0d8b5bdd9cd.png)

## 2018: BERT & Fine-tuned NLP Models

![image](https://user-images.githubusercontent.com/79494088/137576909-51008d2a-0380-4907-8c48-d7e35994f5fb.png)

## 2019/2020

### BIG Language Models

![image](https://user-images.githubusercontent.com/79494088/137576946-eff929ab-723f-48db-a9c8-fa199f3bbe00.png)

### Self-Supervised Learning

![image](https://user-images.githubusercontent.com/79494088/137576954-fb9453cb-78ea-43b3-8941-32e5d799d87c.png)
