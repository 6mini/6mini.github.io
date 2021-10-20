---
title: '[Deep Learning] Optimizer'
description: Optimizer의 정의와 종류(SGD, Momentum, NAG, Adagrad, Adadelta, RMSprop, Adam, AdaMax, NAdam)
categories:
 - Deep Learning
tags: [Deep Learning, Optimizer]
mathjax: enable
---

# Optimizer

![image](https://user-images.githubusercontent.com/79494088/137847104-fc73e10c-79a2-46ef-b00d-2fce4d5f4bae.png)

- 딥러닝에서 학습속도를 빠르고 안정적이게 하기 위해 손실 함수를 최소화하는 최적의 가중치를 업데이트 하는 방법이다.
- Gradient Descent 시 local optima에 빠지게 되는 문제를 방지하기 위한 여러가지 방법 중 하나이다.
- Hyper parameter 최적화에서 얼마나 진행할지 결정하는 Epoch, 내부 뉴런 수, 일정한 데이터를 버려서 과적합을 막아주는 Dropout 등등 많은 파라미터를 조정하지만 그 중 가장 드라마틱하고 쉽게 바꿔주는 것이다.
- [Keras Optimizers Docs 바로가기](https://keras.io/ko/optimizers/#sgd)

# 종류

![image](https://user-images.githubusercontent.com/79494088/137847104-fc73e10c-79a2-46ef-b00d-2fce4d5f4bae.png)

## SGD(Stochastic Gradient Descent)

$$ \theta_{t+1} = \theta_t - \eta\nabla_{\theta}J(\theta ; x^{(i)},y^{(i)}) $$

- 한 번의 파라미터 업데이트를 위해 하나의 훈련 데이터를 사용한다
- 따라서 SGD는 batch gradient보다 빠르게 업데이트된다는 장점이 있다.
  - Batch Gradient: 기본적인 Gradient descent(데이터 전체를 의미)
- 하지만 목적함수의 Gradient가 하나의 데이터에 의해 결정되다보니 매 업데이트마다 들쭉날쭉한 크기의 Gradient로 파라미터를 업데이트하게 된다.
  - 분산이 큰 Gradient는 SGD가 Local minimum에서 빠져나오게 만들 수도 있지만, 수렴을 방해할 수도 있다.

## Momentum

$$ \theta_{t+1} = \theta_t - v_t $$

- 직역하면 관성으로써 SGD에서 관성을 더하는 방법이다.
- 현재 파라미터를 업데이트해줄 때 이전 Gradient도 계산해 포함시켜 주면서 이 문제를 해결한다.
- 예를 들어 SGD를 미끄럼틀이라고 생각하면, Gradient가 극단적으로 0인 평평한 밑면을 만났다해도 빗면의 Gradient가 남아있어 쭉 갈수 있게 만들어준다.
- 만약 모든 Gradient를 모두 고려해준다면 아주 긴 평평한 땅에서도 SGD는 멈추지 않을 것이다.
- 그래서 이전 Gradient의 영향력을 매 업데이트마다 $\gamma$배 씩 감소시킨다.

$$g_t = \eta\nabla_{\theta_t}(\theta_t)$$

$$g_t = \eta\nabla_{\theta_t}(\theta_t)$$

$$v_2 = g_2 + \gamma g_1$$

$$v_3 = g_3 + \gamma g_2 + \gamma^2 g_1$$

$$v_3 = g_3 + \gamma g_2 + \gamma^2 g_1$$

- $\gamma$값은 주로 0.9를 사용한다.

## NAG(Nesterov Accelerated Gradint)

$$v_t = \gamma v_{t-1} + \eta \nabla_{\theta_t}(\theta_t - \gamma v_{t-1})$$

$$\theta_{t+1} = \theta_t - v_t$$

- 앞을 미리 보고 현재의 관성을 조절하여 업데이트 크기를 바꾸는 방식이다.
- SGD가 관성에 의해 수렴 지점에서 요동치는 것을 방지해준다.

## Adagrad

$$ \theta_{t+1, i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii} + \epsilon}}g_{t,i} $$

- 위 세가지의 공통점은 모든 파라미터에 대해 같은 learning rate를 적용하여 업데이트를 한다는 점이다.
  - 문제점: 각 파라미터의 업데이트 빈도 수에 따라 업데이트 크기를 다르게 해줘야 하는데, 앞의 세가지 방법은 이를 반영하지 못한다.
- 이름의 Ada라는 단어는 Adaptive(상황에 맞게 변화하는)의 준말이다.
  - (어뎁티브 그레드 메서드라고 부르는게 더 있어 보인다.)
- 파라미터마다 지금까지 얼마나 업데이트 됐는 지 알기 위해 Adagrad는 parameter의 이전 Gradient를 저장함으로 앞의 문제점을 해결한다.
- 하지만 $t$가 증가하면서 $G_{t, ii}$ 값이 점점 커지게 되어 learning rate가 점점 소실되는 문제점이 있다.

## Adadelta, RMSprop
- Adagrad의 learning rate가 점점 소실되는 문제점을 해결하려는 방식이다.
- Adadelta

$$ \theta_{t+1} = \theta_t + \Delta\theta_t $$

- RMSprop

$$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2] + \epsilon}}g_t $$

## Adam(Adaptive Moment Estimation)

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v_t}}+\epsilon}\hat{m_t}$$

- Adagrad, Adadelta, RMSprop 처럼 각 파라미터마다 다른 크기의 업데이트를 적용하는 방법이다.
- Adadelta에서 사용한 decaying average of squared gradients 뿐만 아니라. decaying average of gradients를 사용한다.

## AdaMax

$$\theta_{t+1} = \theta_t - \frac{\eta}{u_t}\hat{m_t}$$

- Adam의 $v_t$ 텀에 다른 norm을 사용한 방법

## NAdam(Nesterov-accelerated Adaptive Memoment Adam)

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v_t}}+\epsilon}(\beta_1\hat{m}_{t} + \frac{(1-\beta_1)g_t}{1-\beta_1^t})$$

- NAG와 Adam을 섞은 방법이다.

# 결론
- SGD부터 NAdam까지 모든 알고리즘을 살펴보았는데, 각 알고리즘은 이전 알고리즘의 한계점을 보완해가며 발전해갔다.
- SGD는 업데이트 한 번에 데이터 하나를 사용하여 Batch GD의 시간 문제를 해결했다.
- Momentum은 SGD의 작은 gradient 때문에 작은 언덕이나 saddle point를 빠져나가지 못하는 것을 momentum을 도입하여 해결했다.
- NAG는 다음 step의 gradient를 먼저 살펴보고 Momentum을 조절하여 minimum에 안정적으로 들어갈 수 있게 했다.
- Adagrad는 업데이트 빈도가 다른 파라미터에 대해서도 같은 비율로 업데이트하는 것을 이전 gradient들의 합을 기억함으로써 문제를 해결하였다.
- RMSProp과 Adadelta는 Adagrad의 learning rate가 점점 소실되는 것을 gradient의 2차 모먼트를 통해 보완했다.
- Adam은 RMSProp에 1차 모먼트를 도입하여 RMSProp과 Momentum을 합친 효과를 볼 수 있었다.
- AdaMax는 Adam의 2차 모먼트에 있는 gradient의 norm을 max norm으로 바꿔주었다.
- 마지막으로 NAdam은 ADAM에 NAG를 더해주어서 momentum을 보완해주었다.
- 모르겠으면 Adam!

# Refferance
- [Keras Optimizers Documentation](https://keras.io/ko/optimizers/#sgd)
- [[딥러닝] 딥러닝 최적화 알고리즘 알고 쓰자. 딥러닝 옵티마이저(optimizer) 총정리](https://hiddenbeginner.github.io/deeplearning/2019/09/22/optimization_algorithms_in_deep_learning.html)
- [Optimizer](http://www.incodom.kr/Optimizer)
- [[머신러닝] 옵티마이저(Optimizer)란?](https://needjarvis.tistory.com/685)