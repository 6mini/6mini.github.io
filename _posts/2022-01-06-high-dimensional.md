---
title: '[선형대수] 고차원의 문제와 PCA(Principal Component Analysis)'
description: 벡터 변환의 목적과 사용 예시. 고유값과 고유벡터. 데이터의 피쳐 수가 큰 경우의 문제점과 핸들링하기 위한 방법. PCA의 목적과 기본 원리 및 특징
categories:
 - Mathematics
tags: [선형대수, PCA, 고유값, 고유벡터]
mathjax: enable
---

# 벡터 변환(Vector Transformation)

![](https://user-images.githubusercontent.com/6457691/89977531-4a73b400-dca6-11ea-9f43-f0c1f124b70b.jpg){: width="50%"}

![image](https://user-images.githubusercontent.com/79494088/148360092-5de6c698-f7e4-4e7e-a388-e3668024a96a.png){: width="50%"}

- $\mathbb{R}^2$ 공간에서 벡터를 변환할 것이다.
- 여기서의 변환, 즉 선형 변환(Linear Transformation)은 임이의 두 벡터를 더하거나 혹은 스칼라 값을 곱하는 것을 의미한다.
- 전 포스팅의 선형 투영(Linear Projections)도 일종의 벡터 변환이다.

$$
T(u+v) = T(u)+T(v)
\\
T(cu) = cT(u)
$$

## 매트릭스-벡터의 곱
- '`f` 라는 트랜스포메이션(transformation)을 사용하여 임의의 벡터 `[x1, x2]`에 대해 `[2x1 + x2, x1 -3x2 ]`로 변환한다.'를 아래와 같이 표현할 수 있다.

$$
f(\begin{bmatrix}x_1 \\ x_2 \end{bmatrix}) = \begin{bmatrix} 2x_1 + x_2 \\ x_1 -3x_2 \\  \end{bmatrix}
$$

- 여기서 원래의 벡터 `[x1, x2]`는 유닛 벡터를 이용하여 아래처럼 분리할 수 있다.

$$x_1 \cdot \hat{i} + x_2 \cdot \hat{j}$$

- 분리된 각 유닛 벡터는 트랜스포메이션을 통해 각각,
  - $2x_1$, $x_1$과
  - $x_2$, $-3x_2$라는 결과가 나와야 한다.
- 이를 매트릭스 형태로 합치면,

$$
T = \begin{bmatrix} 2 & 1 \\ 1 & -3 \end{bmatrix}
$$

- 위와 같은 T라는 매트릭스를 얻을 수 있고, 이 매트릭스를 처음 벡터 `[x1, x2]`에 곱했을 경우, 트랜스포메이션이 원하는 대로 이루어진다는 것을 알 수 있다.
- 즉, 임의의 $\mathbb{R}^2$ 벡터를 다른 $\mathbb{R}^2$ 내부의 벡터로 변환하는 과정은, 특정 $T$라는 매트릭스를 곱하는 것과 동일한 과정이다.
- 새로운 벡터 $(3, 4)$에 대하여 동일한 필터로 변환하는 경우, 방금 구한 $T$라는 매트릭스에 곱하는 것으로 쉽게 할 수 있다.

$$
\begin{bmatrix} 2 & 1 \\ 1 & -3 \end{bmatrix}\begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 10 \\ -9 \end{bmatrix}
$$

- 벡터 변환은 곱하고 더하는 것으로만 이루어진 선형 변환이기 때문에, 매크릭스와 벡터의 곱으로 표현할 수 있다.

```py
import matplotlib.pyplot as plt
import numpy as np

input_vector = np.array([3, 4])

transform_matrix = np.array([[2, 1], [1, -3]])

output_vector = np.matmul(transform_matrix, input_vector)

print(output_vector)

plt.arrow(0, 0, input_vector[0], input_vector[1], head_width = .05, head_length = .05, color ='#d63031')
plt.arrow(0, 0, output_vector[0], output_vector[1], head_width = .05, head_length = .05, color ='#0984e3')
plt.xlim(0, 12)
plt.ylim(-10, 5)
plt.title("Transformed Vector")
plt.show()
'''
[10, -9] 
'''
```

![image](https://user-images.githubusercontent.com/79494088/148348708-926f7880-1d84-44f7-a87b-94169954a72a.png)

## 고유벡터(Eigenvector)

![image](https://user-images.githubusercontent.com/79494088/148348848-f7e52c24-73d5-4709-9c8a-eca33fbef1aa.png){: width="70%"}

- 트랜스포메이션은 매트릭스를 곱하는 것을 통해 벡터(데이터)를 다른 위치로 옮긴다는 의미를 갖고 있다.
- $\mathbb{R^3}$ 공간에서의 트랜스포메이션을 사용해보면, 위의 회전하는 지구본은 $\mathbb{R^3}$ 공간에서의 임의의 위치에서 다른 위치로 옮겨진다는 것을 설명하고 있다.
- $\mathbb{R^3}$ 공간이 회전할 때, 위치에 따라 변화하는 정도가 다르다는 것을 알 수 있다.
  - 적도 부근의 점이 변화되는 거리와, 극지방에 있는 점의 위치 변화의 크기는 다르다.
  - 회전축으로 가까이 갈수록/멀어질수록 더욱 명확해지며, 정확하게 회전축에 위치해있는 경우, 트랜스포메이션을 통해 위치가 변하지 않는다.
- 이러한 **트랜스포메이션에 영향을 받지않는 회전축을 공간의 고유벡터(Eigenvector)**라고 부른다.

## 고유값(Eigenvalue)
- 위의 고유벡터는 주어진 트랜스포메이션에 대해 크기만 변하고 방향은 변화하지 않는 벡터이다.
- 여기서 **변화하는 크기는 결국 스칼라 값으로 변화할 수 밖에 없는데, 이 특정 스칼라 값을 고유값(Eigenvalue)**라고 한다.
- 고유벡터와 고유값은 항상 쌍을 이룬다.

$$ T \cdot v = v' = \lambda \cdot v $$

$$
\begin{bmatrix} a & b \\ c & d \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} ax+by \\ cx+dy \end{bmatrix} = \lambda \begin{bmatrix} x \\ y \end{bmatrix}
$$

- 예제

$$
\begin{bmatrix} 4 & 2 \\ 2 & 4 \end{bmatrix}\begin{bmatrix} 3 \\ -3 \end{bmatrix} = \begin{bmatrix} 6 \\ -6 \end{bmatrix} = 2 \begin{bmatrix} 3 \\ -3 \end{bmatrix}
$$

### 고유값의 표기
- $\lambda$로 표현한다.

$$
T(v) = \lambda v
$$

### 고유값을 배우는 이유
- 벡터 변환은 결국 궁극적으로 '데이터를 변환한다'는 큰 목적의 단계 중 하나이다.
- 예를 들어,

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

- 라는 데이터는,

$$\begin{bmatrix} 5 \\ 6 \end{bmatrix}$$

- 로 변환할 수 있고,

$$\begin{bmatrix} 7 \\ 8 \end{bmatrix}$$

- 로 변환할 수도 있고, 목적에 따라 거의 무한한 방법으로 트랜스포메이션할 수 있는데, 그 중 어떤 목적으로 어떤 변환을 하냐에 따라 고유값이 하나의 선택지가 된다.
- 고유값은 고차원의 문제를 해결하려는 목적에 쓰인다.

# 고차원의 문제(The Curse of Dimensionality)
- 고차원의 문제란 피쳐의 수가 많은 데이터셋을 모델링하거나 분석할 때 생기는 여러가지 문제점을 의미한다.

## 차원(Dimension)
- 임의의 50개 수로 이뤄진 데이터셋을 가정한다.

```py
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 50개 데이터 생성 후 데이터 프레임에 저장
N = 50
x = np.random.rand(N)*100

data = {"x": x}
df = pd.DataFrame(data)
df.head()
```

![image](https://user-images.githubusercontent.com/79494088/148351378-287a11a3-b466-46fa-922e-ca2e058e94fa.png)

```py
# 선에 데이터 표기

def setup(ax):
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which = 'major', width = 1)
    ax.tick_params(which = 'major', length = 5)
    ax.tick_params(which = 'minor', width = .75)
    ax.tick_params(which = 'minor', length = 2.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.patch.set_alpha(0)
    
plt.figure(figsize=(8, 6))
n = 8

df['y'] = pd.Series(list(np.zeros(50)))

ax = plt.subplot(n, 1, 2)
setup(ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.text(0, 0.5, "Number Line", fontsize = 14, transform = ax.transAxes)

plt.subplots_adjust(left = .05, right = .95, bottom = .05, top = 1.05)
plt.scatter(df.x, df.y, alpha = .5)
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/148351437-5c3a1663-6510-4777-9fba-6348f971689c.png)

### 2D
- 이번엔 피쳐가 2개이다.

```py
# 임의의 50개 feature값을 생성
df['y'] = pd.Series(list(np.random.rand(N)*100))


plt.scatter(df['x'], df['y'], alpha = .5)
plt.title("A Better Use of a 2D Graph")
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/148351615-3bd8b417-9d11-4a60-b689-cd5b78f15cb8.png)

### 3D

```py
from mpl_toolkits.mplot3d import Axes3D

# z 값을 추가
df['z'] = pd.Series(list(np.random.rand(N)*100))

threedee = plt.figure().gca(projection = '3d')
threedee.scatter(df['x'], df['y'], df['z'])
threedee.set_xlabel('X')
threedee.set_ylabel('Y')
threedee.set_zlabel('Z')
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/148351685-5abfd8a9-6a60-4158-bef4-84a36de0f2dd.png)

### 4D ~ 20D

![image](https://user-images.githubusercontent.com/79494088/148351824-cda6f449-9f65-460a-8ff2-92a69eacc093.png){: width="70%"}

- 사람의 뇌는 3차원 이상의 정보를 공간적으로 다루는 것이 거의 불가능하다.
- 다시 말해 이는 여러 차원의 데이터셋을 다루는 데 있어 큰 이슈가 된다.

## 복잡한 시각화

![image](https://user-images.githubusercontent.com/79494088/148352409-97065142-4747-4166-95a0-40e0847241a8.png)

- 차원이 높아질수록 의미없는 Plot이 많아진다.

## 추가 피쳐 사용과 결과물
- 데이터셋에서 인사이트를 찾기 위해 쓰이는 모든 피쳐가 동일하게 중요하지 않다.
- 피쳐를 추가로 사용하는 것이 실제로 얼마나 의미있게 더 좋은 결과를 모델링 하게 되는 지는 고민해야 할 문제이다.

{% include ad.html %}

## 과적합(Overfitting)

![](https://user-images.githubusercontent.com/6457691/89990876-f4126f80-dcbd-11ea-9046-8a4d7181d2ea.jpg){: width="50%"}

- 샘플 수에 비해 피쳐 수가 많은 경우 과적합의 문제 또한 발생한다.

# 차원 축소(Dimension Reduction)
- 데이터의 시각화나 탐색이 어려워지는 것 뿐 아니라 모델링에서의 과적합 이슈를 포함하는 등 빅데이터인 데이터셋의 피쳐가 많으면 많을수록 이로 인해 발생하는 문제는 점점 많아진다.
- 만약 데이터의 적절한 처리를 통해 충분한 의미를 유지하면서 더 작은 부분만 선택해야한다.
- 머신러닝에서는 이를 위한 다양한 차원 축소 기술이 이미 연구되어있고, 지금도 연구중이다.

## 피쳐 선택(Featrue Selection)
- 분석해야할 데이터셋에 100개의 피쳐가 있다고 가정한다.
- 100개의 피쳐를 전부 사용하는 대신, 데이터셋에 제일 다양하게 분포되어있는 하나의 피쳐를 사용하는 것이다.
- 이처럼 피쳐 선택이란 데이터셋에서 덜 중요한 피쳐를 제거하는 방법을 의미한다.

## 피쳐 추출(Feature Extraction)
- 기존의 피쳐 혹은 그들을 바탕으로 조합된 피쳐를 사용하는 것으로 PCA가 한 예시이다.

### 피쳐 선택과 추출의 차이

![image](https://user-images.githubusercontent.com/79494088/148360356-c55a0c1d-1eb6-4e42-be7f-cb7f65f8d8f1.png){: width="50%"}

- 피쳐 선택
  - 장점: 선택된 피쳐의 해석이 쉽다.
  - 단점: 피쳐들간의 연관성을 고려해야한다.
  - 예시: `LASSO`, `Genetic algorithm` 등

- 피쳐 추출
  - 장점: 피쳐들간의 연관성이 고려되고 피쳐 수를 많이 줄일 수 있다.
  - 단점: 피쳐 해석이 어렵다.
  - 예시: `PCA`, `Auto-encoder` 등

# PCA(Principal Component Analysis)
- 고차원의 데이터를 효과적으로 분석하기 위한 기법이다.
- 낮은 차원으로 차원을 축소한다.
- 고차원의 데이터를 효과적으로 시각화하고 클러스터링(Clustering)한다.
- 원래 고차원 데이터의 정보(분산)을 최대한 유지하는 벡터를 찾고, 해당 벡터에 대해 데이터를 선형 투영(Linear Projection)한다.

```py
import pandas as pd
import matplotlib.pyplot as plt

x = [-2.2, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 2.2]
y = [0, .5, -.5, .8, -.8, .9, -.9, .8, -.8, .5, -.5, 0]

df = pd.DataFrame({"x": x, "y": y})

print('variance of X : ' + str(np.var(x)))
print('variance of Y : ' + str(np.var(y)))

plt.scatter(df['x'], df['y'])
plt.arrow(-3, 0, 6, 0, head_width = .05, head_length = .05, color ='#d63031')
plt.arrow(0, -1, 0, 6, head_width = .05, head_length = .05, color ='#00b894');
'''
variance of X : 2.473333333333333
variance of Y : 0.4316666666666668
'''
```

![image](https://user-images.githubusercontent.com/79494088/148357990-7a3285bd-b672-431d-9ac2-ef6635338d13.png)

- 위 스캐터 플롯(scatter plot)에 그려진 각 포인트들을 2차원의 데이터셋을 의미한다고 가정한다.
- 2개의 피쳐 중 1개만을 분석에 사용해야하여 차원 축소를 하려한다면 어느 피쳐를 사용해야 할까?

```py
import math

df["x_rotate"] = df.apply(lambda x: (x.x+x.y)/math.sqrt(2), axis=1)
df["y_rotate"] = df.apply(lambda x: (x.y-x.x)/math.sqrt(2), axis=1)

plt.scatter(df['x_rotate'], df['y_rotate'])
plt.arrow(-2, 2, 6, -6, head_width = .05, head_length = .05, color ='#d63031')
plt.arrow(-2, -2, 6, 6, head_width = .05, head_length = .05, color ='#00b894');
```

![image](https://user-images.githubusercontent.com/79494088/148358264-1853496e-d628-482a-b784-fb80688d990b.png)

- 만약 데이터가 위 x, y축에 평행하지 않다면 어떻게 해야할까?
- 정답은 x도 y도 아닌 데이터의 흩어진 정도를 가장 크게 하는 벡터 축이 될 것이다.

## PCA 프로세스

![image](https://user-images.githubusercontent.com/79494088/148358421-2ab32187-9d71-4e34-939d-cb2e36405e9b.png){: width="70%"}

- 다차원의 데이터를 시각화하기 위해 2차원으로 축소하는데, 제일 정보 손실이 적은 2차원을 고른다.

### 1. 데이터 준비

```py
import numpy as np

X = np.array([ 
              [0.2, 5.6, 3.56], 
              [0.45, 5.89, 2.4],
              [0.33, 6.37, 1.95],
              [0.54, 7.9, 1.32],
              [0.77, 7.87, 0.98]
])
print("Data: ", X)
'''
Data:  [[0.2  5.6  3.56]
 [0.45 5.89 2.4 ]
 [0.33 6.37 1.95]
 [0.54 7.9  1.32]
 [0.77 7.87 0.98]]
'''
```

### 2. 각 열에 대해 평균을 빼고 표준편차로 나눠 노멀라이즈(Normalize)

```py
standardized_data = ( X - np.mean(X, axis = 0) ) / np.std(X, ddof = 1, axis = 0)
print("\n Standardized Data: \n", standardized_data)
'''
 Standardized Data: 
 [[-1.19298785 -1.0299848   1.5011907 ]
 [-0.03699187 -0.76471341  0.35403575]
 [-0.59186994 -0.32564351 -0.09098125]
 [ 0.37916668  1.07389179 -0.71400506]
 [ 1.44268298  1.04644992 -1.05024014]]
'''
```

### 3. $Z$의 분산-공분산 매트릭스 계산
- $Z^{T}Z$를 통해 계산 할 수 있다.

```py
covariance_matrix = np.cov(standardized_data.T)
print("\n Covariance Matrix: \n", covariance_matrix)
'''
 Covariance Matrix: 
 [[ 1.          0.84166641 -0.88401004]
 [ 0.84166641  1.         -0.91327498]
 [-0.88401004 -0.91327498  1.        ]]
'''
```

### 4. 분산-공분산 매트릭스의 고유벡터와 고유값 계산

```py
values, vectors = np.linalg.eig(covariance_matrix)
print("\n Eigenvalues: \n", values)
print("\n Eigenvectors: \n", vectors)
'''
 Eigenvalues: 
 [2.75962684 0.1618075  0.07856566]

 Eigenvectors: 
 [[ 0.56991376  0.77982119  0.25899269]
 [ 0.57650106 -0.60406359  0.55023059]
 [-0.58552953  0.16427443  0.7938319 ]]
'''
```

### 5. 데이터를 고유벡터에 프로젝션(projection)

```py
Z = np.matmul(standardized_data, vectors)

print("\n Projected Data: \n", Z)
'''
 Projected Data: 
 [[-2.15267901 -0.06153364  0.31598878]
 [-0.66923865  0.4912475  -0.14930446]
 [-0.47177644 -0.27978923 -0.40469283]
 [ 1.25326312 -0.47030949  0.12228952]
 [ 2.04043099  0.32038486  0.11571899]]
'''
```

- PCA는 고차원의 데이터를 분산을 유지하는 축(`PC`)을 기반으로 데이터를 변환한 것이며, 해당 PC들 중 일부를 사용하는 것으로 차원 축소를 할 수 있다.
- 즉,

|$x_1$|$x_2$|$x_3$|
|:-:|:-:|:-:|
|0.2| 5.6| 3.56|
|0.45|5.89|2.4|
|0.33|6.37|1.95|
|0.54|7.9|1.32|
|0.77|7.87|0.98|

- 라는 데이터가,

|$pc_1$|$pc_2$|$pc_3$|
|:-:|:-:|:-:|
|-2.1527|-0.0615|0.3160|
|-0.6692|0.4912|-0.1493|
|-0.4718|-0.2798|-0.4047|
|1.2533|-0.4703|0.1223|
|2.0404|0.3204|0.1157|

- 라는 데이터로 변환이 되었고 이 중,

|$pc_1$|$pc_2$|
|:-:|:-:|
|-2.1527|-0.0615|
|-0.6692|0.4912|
|-0.4718|-0.2798|
|1.2533|-0.4703|
|2.0404|0.3204|

- 를 사용할 경우 2차원으로 축소 했다는 의미가 있다.

## 시각화

```py
df = pd.DataFrame({"x1": [0.2, 0.45, 0.33, 0.54, 0.77] , "x2": [5.6, 5.89, 6.37, 7.9, 7.87], 'x3': [3.56, 2.4, 1.95, 1.32, 0.98]})

threedee = plt.figure().gca(projection = '3d')
threedee.scatter(df['x1'], df['x2'], df['x3'])
threedee.set_xlabel('x1')
threedee.set_ylabel('x2')
threedee.set_zlabel('x3')
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/148359250-3622e74e-6613-4afa-9343-f27ff5716044.png)

```py
df = pd.DataFrame({"pc1": [-2.1527, -0.6692, -0.4718, 1.2533, 2.0404], "pc2": [-0.0616, 0.4912, -0.2798, -0.4703, 0.3204]})
plt.scatter(df['pc1'], df['pc2'])
plt.title("Data After PCA")
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/148359313-39caa968-26d2-4c68-9209-f38f4de80b3d.png)

## 라이브러리

```py
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA

print("Data: \n", X)

scaler = StandardScaler()
Z = scaler.fit_transform(X)
print("\n Standardized Data: \n", Z)

pca = PCA(2)

pca.fit(Z)

print("\n Eigenvectors: \n", pca.components_)
print("\n Eigenvalues: \n",pca.explained_variance_)

B = pca.transform(Z)
print("\n Projected Data: \n", B)
'''
Data: 
 [[0.2  5.6  3.56]
 [0.45 5.89 2.4 ]
 [0.33 6.37 1.95]
 [0.54 7.9  1.32]
 [0.77 7.87 0.98]]

 Standardized Data: 
 [[-1.33380097 -1.15155802  1.67838223]
 [-0.04135817 -0.85497558  0.395824  ]
 [-0.66173071 -0.36408051 -0.10172014]
 [ 0.42392124  1.20064752 -0.79828193]
 [ 1.61296861  1.16996658 -1.17420417]]

 Eigenvectors: 
 [[-0.13020816 -0.73000041  0.67092863]
 [-0.08905388  0.68256517  0.72537866]]

 Eigenvalues: 
 [2.15851707 0.09625196]

 Projected Data: 
 [[ 1.87404384  0.35553233]
 [ 0.85151446 -0.31022649]
 [ 0.21482136 -0.29832914]
 [-1.35210803  0.27030569]
 [-1.58827163 -0.0172824 ]]
'''
```

## PCA의 특징
- 데이터에 대해 독립적인 축을 찾는데 사용할 수 있다.
- 데이터의 분포가 정규성을 띄지 않는 경우 적용이 어렵다.
  - 이 경우엔 커널 PCA를 사용할 수 있다.
- 분류/예측 문제에 대해 데이터의 라벨을 고려하지 않기 때문에 효과적 분리가 어렵다.
  - 이 경우엔 PLS를 사용할 수 있다.