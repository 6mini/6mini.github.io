---
title: '[Deep Learning] Tensorflow & Keras'
description: 모델 아키텍쳐 선택. 가중치의 규제 전략. 다양한 활성함수를 사용함에 발생하는 trade-off에 대한 논의
categories:
 - Deep Learning
tags: [Deep Learning, Regularization, Loss Function, 규제방법]
mathjax: enable
---

# Warm-Up

## Review
- 신경망의 동작원리 (Note1)
  * 데이터 전처리 및 입력
  * 모델 제작 및 가중치 초기화
  * 모델에 데이터를 넣고 출력값을 얻음
  * 출력값과 레이블(정답지)과 비교 후 Loss 계산
  * Loss를 반영하여 가중치 업데이트 -> 역전파(BackPropagation) + 경사하강법(Gradient Descent)
- 역전파 원리 및 실습
  * Loss function의 계산방식
  * Stochastic Gradient Descent 방법
  * 경사하강법의 변형들(Adam)
  * 2x2x2 neural network의 역전파 수학식
- Fashion MNIST 실습

# Library

```py
from google.colab import files

# 폐암 수술 환자의 특정기간 생존 데이터
# 속성(정보)은 종양의 유형, 폐활량, 호흡곤란 여부, 기침, 흡연, 천식여부 등의 17가지 환자 상태. 수술 후 생존(1), 사망(0) 
my_data = "https://ds-lecture-data.s3.ap-northeast-2.amazonaws.com/everydeep/ThoraricSurgery.csv"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.layers as Layer
import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)

Data_set = np.loadtxt(my_data, delimiter=",") 

Data_set = pd.read_csv(my_data, header=None)

Data_set = np.loadtxt(my_data, delimiter=",") 
# 환자의 기록과 수술 결과를 X와 Y로 구분하여 저장
X = Data_set[:,0:17]
Y = Data_set[:,17]

# 딥러닝 구조를 결정
model = Sequential([
    Dense(30, activation='relu'),
    Layer.Dropout(0.5),
    Dense(30, activation='relu'),
    Dense(1, activation='sigmoid') # 분류할 방법에 따라 개수 조정
])
# 딥러닝 실행
model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy']) # mean_squared_error # binary_crossentropy # mean_absolute_error # poisson
history = model.fit(X, Y, epochs=30, batch_size=30)
'''
Epoch 1/30
16/16 [==============================] - 1s 2ms/step - loss: 0.1923 - accuracy: 0.5833
Epoch 2/30
16/16 [==============================] - 0s 1ms/step - loss: 0.0986 - accuracy: 0.7804
Epoch 3/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0887 - accuracy: 0.8106
Epoch 4/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0765 - accuracy: 0.8378
Epoch 5/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0742 - accuracy: 0.8453
Epoch 6/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0683 - accuracy: 0.8559
Epoch 7/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0735 - accuracy: 0.8452
Epoch 8/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0753 - accuracy: 0.8432
Epoch 9/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0669 - accuracy: 0.8616
Epoch 10/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0632 - accuracy: 0.8710
Epoch 11/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0731 - accuracy: 0.8474
Epoch 12/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0805 - accuracy: 0.8310
Epoch 13/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0854 - accuracy: 0.8202
Epoch 14/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0758 - accuracy: 0.8424
Epoch 15/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0763 - accuracy: 0.8415
Epoch 16/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0669 - accuracy: 0.8595
Epoch 17/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0668 - accuracy: 0.8570
Epoch 18/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0704 - accuracy: 0.8547
Epoch 19/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0764 - accuracy: 0.8424
Epoch 20/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0664 - accuracy: 0.8596
Epoch 21/30
16/16 [==============================] - 0s 3ms/step - loss: 0.0734 - accuracy: 0.8475
Epoch 22/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0684 - accuracy: 0.8580
Epoch 23/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0703 - accuracy: 0.8541
Epoch 24/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0741 - accuracy: 0.8449
Epoch 25/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0787 - accuracy: 0.8356
Epoch 26/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0695 - accuracy: 0.8552
Epoch 27/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0729 - accuracy: 0.8465
Epoch 28/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0749 - accuracy: 0.8437
Epoch 29/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0598 - accuracy: 0.8733
Epoch 30/30
16/16 [==============================] - 0s 2ms/step - loss: 0.0544 - accuracy: 0.8865
'''
```

## Loss Function

### 평균제곱계열
- mean_squared_error (MSE) = $\frac{1}{n}\sum_{i=1}^{n}(y_{i} - \hat{y_{i}})^{2}$
- RMSE (Root Mean Squared Error) = 
$\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_{i} - \hat{y_{i}})^{2}}$
- mean_absolute_error (MAE) = $\frac{1}{n}\sum_{i=1}^{n}\left \vert y_{i} - \hat{y_{i}} \right \vert$
- R-Squared (coefficient of determination) = $1 - \frac{\sum_{i=1}^{n}(y_{i} - \hat{y_{i}})^{2}}{\sum_{i=1}^{n}(y_{i} - \bar{y_{i}})^{2}} = 1 - \frac{SSE}{SST} = \frac {SSR}{SST}$
  -  SSE, SST, SSR: Sum of Squared `Error`, `Total`, `Regression`($\sum_{i=1}^{n}(\hat{y_{i}} - \bar{y_{i}})^{2}$)
- mean_absolute_percentage_error = $ \frac {1}{n}\sum _{i=1}^{n}\left\vert{\frac {y_{t}-\hat{y_{i}}}{y_{i}}}\right\vert $
- mean_squared_logarithmic_error = $\frac{1}{n} \sum_{i=1}^n (\log(\hat{y_i} + 1) - \log(y_i+1))^2 $

### 엔트로피계열
- binary_crossentropy = $ -\sum_{c=1}^{C} q(y_c) log(q(y_c)), \hspace{2em} q(y_c) \in (1, -1)$
- categorical_crossentropy = $ -\sum_{c=1}^{C} q(y_c)log(q(y_c)) $

# Regularization Strategies

## Overfitting

### EarlyStopping
- Loss가 여러 Epoch 동안 감소하지 않으면 Overfitting으로 간주하여 학습을 중단

![image](https://user-images.githubusercontent.com/79494088/137959746-dd71e8af-9f3d-4eaf-b597-cc6119d6c79d.png)

### Weight Decay

![image](https://user-images.githubusercontent.com/79494088/137959805-ea1143ea-a9f0-43ac-907a-29a064c7decd.png)

### Weight Constraint / Weight Decusion / Weight Restriction

### Dropout
- 신경망의 각 레이어 노드에서 학습할 때마다 일부 노드를 사용하지 않고 학습을 진행하는 방법
- 노드 간의 연결 자체를 사용하지 않도록 만들면서 하나의 모델을 여러가지 방법으로 학습
- 지정한 비율의 뉴런을 제거
- 테스트 시에는 모든 뉴런을 사용하기 때문에 여러 Network를 Ensemble 하는 효과를 가진다.


![image](https://user-images.githubusercontent.com/79494088/137960075-430c17c8-9a09-4087-acf9-43ce8ccbfdfb.png)

### Batch Normalization
- 중간 Feature를 그대로 사용하지 않고 변형하여 학습

![image](https://user-images.githubusercontent.com/79494088/137960673-98f509aa-45aa-43ac-9482-5a13a9a1cac6.png)

```py
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.
X_test = X_test /255.

# 기본적인 신경망
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import keras, os

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam'
             , loss='sparse_categorical_crossentropy'
             , metrics=['accuracy'])
model.summary()
# 총 7850 parameters (10 bias)
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                7850      
=================================================================
Total params: 7,850
Trainable params: 7,850
Non-trainable params: 0
_________________________________________________________________
'''


batch_size = 30
epochs_max = 1

# 학습시킨 데이터 저장
checkpoint_filepath = "FMbest.hdf5"

# overfitting을 방지하기 위해서 학습 중 early stop을 수행
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

# Validation Set을 기준으로 가장 최적의 모델
save_best = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch', options=None)

# 모델 학습 코드 + early stop + Best model
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs_max, verbose=1, 
          validation_data=(X_test,y_test), 
          callbacks=[early_stop, save_best])
'''
2000/2000 [==============================] - 4s 2ms/step - loss: 0.7801 - accuracy: 0.7314 - val_loss: 0.5107 - val_accuracy: 0.8221

Epoch 00001: val_loss improved from inf to 0.51069, saving model to FMbest.hdf5
<tensorflow.python.keras.callbacks.History at 0x7f618d2dbb50>
'''


model.predict(X_test[0:1])
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
'''
313/313 - 0s - loss: 0.5107 - accuracy: 0.8221
'''
```

# 규제방법 구현

```py
Dense(64, input_dim=64,
          kernel_regularizer=regularizers.l2(0.01),
          activity_regularizer=regularizers.l1(0.01))
```

## Weight Decay

$L(\theta_w) = {1 \over 2} \Sigma_i (output_i - target_i)^2 + \lambda \vert\theta_w\vert$ 

$L(\theta_w) = {1 \over 2} \Sigma_i (output_i - target_i)^2 + \lambda \vert\vert\theta_w\vert\vert_2$ 

- 가중치를 감소시키는 기술로써, 애초에 큰 가중치를 갖지 못하게 만드는 기술

![image](https://user-images.githubusercontent.com/79494088/138018014-8a27a24c-1593-41c4-92b6-37d4c61d2ea8.png)

```py
# Weight Decay를 전체적으로 반영한 예시 코드
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import regularizers

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(64, 
            kernel_regularizer=regularizers.l2(0.01),    # L2 norm regularization
            activity_regularizer=regularizers.l1(0.01)), # L1 norm regularization    
    Dense(10, activation='softmax')
])
# 업데이트 방식을 설정
model.compile(optimizer='adam'
             , loss='sparse_categorical_crossentropy'
             , metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, batch_size=30, epochs=1, verbose=1, 
          validation_data=(X_test,y_test))
'''
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_1 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 64)                50240     
_________________________________________________________________
dense_5 (Dense)              (None, 10)                650       
=================================================================
Total params: 50,890
Trainable params: 50,890
Non-trainable params: 0
_________________________________________________________________
2000/2000 [==============================] - 4s 2ms/step - loss: 1.3607 - accuracy: 0.7677 - val_loss: 0.8033 - val_accuracy: 0.8060
<tensorflow.python.keras.callbacks.History at 0x7f617e54dd10>
'''
```

- [Overfitting](https://towardsdatascience.com/over-fitting-and-regularization-64d16100f45c)

- [L2 regularization](https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization)

- [L1 vs L2 한국어 설명](https://light-tree.tistory.com/125)

### L1, L2

#### Norm
- 벡터의 크기를 측정하는 방법이다.
- 두 벡터 사이의 거리를 측정하는 방법이기도 하다.

#### L1 Norm

![image](https://user-images.githubusercontent.com/79494088/138432640-f2026e67-2133-4fe8-8c59-6a7c1804c5a0.png)

- 벡터 p, q 각 원소의 차이의 절대값의 합이다.

#### L2 Norm

![image](https://user-images.githubusercontent.com/79494088/138432713-ac877fc9-d39d-4e92-aef9-64b2c8b503a1.png)

- 벡터 p, q의 유클리디안 거리이다.
- q가 원점이라면 벡터 p, q의 L2 Norm은 벡터 p의 원점으로부터의 직선거리이다.

#### L1 Norm과 L2 Norm의 차이

![image](https://user-images.githubusercontent.com/79494088/138432934-86f4010c-6927-4067-9c95-644c2c2e00bf.png)

- 검정색 두 점사이의 L1 Norm은 빨간색, 파란색, 노란색 선으로 표현될 수 있고, L2 Norm은 오직 초록색선으로만 표현될 수 있다.
- **<font color='red'>L1 Norm은 여러가지 path를 가지지만 L2 Norm은 Unique shortest path를 가진다.</font>**

#### L1 Loss

![image](https://user-images.githubusercontent.com/79494088/138433356-67d3dbed-3268-4fea-9381-2154311f48b2.png)

- 실제값과 예측치 사이의 차이 값의 절대값을 구하고 그 오차의 합을 L1 Loss라고 한다.

#### L2 Loss

![image](https://user-images.githubusercontent.com/79494088/138433490-6e7d923b-88a5-4ae7-aa87-6b5fc48cc1a5.png)

- 오차의 제곱의 합으로 정의된다.

#### L1 Loss와 L2 Loss의 차이
- L2 Loss는 직관적으로 오차의 제곱을 더하기 때문에 Outlier에 더 큰 영향을 받는다.
- L1 Loss가 L2 Loss에 비해 Outlier에 대하여 더 Robust하다.
- Outlier가 적당히 무시되길 원한다면 L1 Loss를 사용하고, Outlier의 등장에 신경써야 하는 경우라면 L2 Loss를 사용하는 것이 좋다.
- L1 Loss는 0인 지점에서 미분이 불가능하다는 단점을 갖고 있다.

#### L1 Regularization

![image](https://user-images.githubusercontent.com/79494088/138434007-f3d212a6-3884-4a4c-9d78-c611f32c2b01.png)

- 가장 중요한 것은 cost func에 가중치의 절대값을 더해준다는 것이다.
- 기존의 cost func에 가중치의 크기가 포함되면서 가중치가 너무 크지 않은 방향으로 학습되도록 한다.
- 이때 ${lambda}$는 Learning rate 같은 상수로 0에 가까울수록 정규화의 효과는 없어진다.

#### L2 Regularization

![image](https://user-images.githubusercontent.com/79494088/138434343-a877b08f-5d1b-498c-9212-ab226e0c2317.png)

- 기존의 cost func에 가중치의 제곱을 포함하여 더함으로써 L1 Regularization과 마찬가지로 가중치가 너무 크지 않은 방향으로 학습되게 되며 이를 Weight decay라고도 한다.
- L2 Regularization을 사용하는 Regression model을 Ridge Regression이라고 부른다.

#### L1 Regularization, L2 Regularization의 차이와 선택 기준
- 가중치 w가 작아지도록 학습한다는 것은 결국 Local noise에 영향을 덜 받도록 하겠다는 것이며 이는 Outlier의 영향을 더 적게 받도록 하겠다는 것이다.

![image](https://user-images.githubusercontent.com/79494088/138435768-2d51d5d3-71f6-41e4-9d6a-b81cc1321a77.png)

- a와 b에 대해서 L1 Norm과 L2 Norm을 계산하면 각각 아래와 같다.

![image](https://user-images.githubusercontent.com/79494088/138435874-9194aaf8-2620-4dd4-a1d2-4e404b2f1e98.png)

![image](https://user-images.githubusercontent.com/79494088/138435917-98ab3a45-a3ed-41d5-be1a-7b7c8e5db866.png)

- L2 Norm은 각각의 벡터에 대해 항상 Unique한 값을 내지만, L1 Norm은 경우에 따라 특정 Feature 없이도 같은 값을 낼 수 있다.

![image](https://user-images.githubusercontent.com/79494088/138436423-ac3f164a-92d0-4d45-b2c1-bba86e349c91.png)

- L1 Norm은 파란색 선 대신 빨간색 선을 사용하여 특정 Feature를 0으로 처리하는 것이 가능하다고 이해할 수 있다.
- L1 Norm은 Feature selection이 가능하고 이런 특징이 L1 Regularization에 동일하게 적용될 수 있는 것이다.
- 이러한 특징 때문에 L1은 Sparse model에 적합하고 convex optimization에 유용하게 쓰인다.

![image](https://user-images.githubusercontent.com/79494088/138436922-e1e71e09-18ac-448e-9d64-35faf072e782.png)

- L1 Regularization의 경우 위 그림처럼 미분 불가능한 점이 있기 때문에 Gradient base learning에는 주의가 필요하다.

## Constraints
- [참고자료](https://keras.io/api/layers/constraints/)
- 물리적으로 Weight의 크기를 제한하는 방법이다.
- Weight 자체를 함수를 이용하여 더 큰 경우는 임의의 값으로 변경해버리는 기술을 사용하게 된다.

```py
# 모델 구성을 확인
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(64, 
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l1(0.01),
            kernel_constraint=MaxNorm(2.)),             ## add constraints
    Dense(10, activation='softmax')
])
# 업데이트 방식을 설정
model.compile(optimizer='adam'
             , loss='sparse_categorical_crossentropy'
             , metrics=['accuracy'])
model.summary()


model.fit(X_train, y_train, batch_size=30, epochs=1, verbose=1, 
          validation_data=(X_test,y_test))
'''
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_2 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 64)                50240     
_________________________________________________________________
dense_7 (Dense)              (None, 10)                650       
=================================================================
Total params: 50,890
Trainable params: 50,890
Non-trainable params: 0
_________________________________________________________________
2000/2000 [==============================] - 5s 2ms/step - loss: 1.3686 - accuracy: 0.7639 - val_loss: 0.7866 - val_accuracy: 0.8135
<tensorflow.python.keras.callbacks.History at 0x7f617d3bf090>
'''
```

## Dropout
- 모델 자체에 Layer를 추가하는 방식으로 진행되는데, 이는 확률적으로 노드 연결을 강제로 끊어주는 역할을 한다.
- 임시로 차단하고 그 연결없이 결과를 예측하도록 하고, 해당 뉴런 없이 학습을 진행하기 때문에 과적합을 어느정도 차단할 수 있다.

```py
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import regularizers
# 모델 구성을 확인
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(64, 
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l1(0.01),
            kernel_constraint=MaxNorm(2.)),             
    Dropout(0.5),# add dropout
    Dense(10, activation='softmax')
])
# 업데이트 방식 설정
model.compile(optimizer='adam'
             , loss='sparse_categorical_crossentropy'
             , metrics=['accuracy'])
model.summary()


model.fit(X_train, y_train, batch_size=30, epochs=1, verbose=1, 
          validation_data=(X_test,y_test))
'''
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_3 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_8 (Dense)              (None, 64)                50240     
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_9 (Dense)              (None, 10)                650       
=================================================================
Total params: 50,890
Trainable params: 50,890
Non-trainable params: 0
_________________________________________________________________
2000/2000 [==============================] - 5s 2ms/step - loss: 1.5921 - accuracy: 0.7224 - val_loss: 0.8783 - val_accuracy: 0.8052
<tensorflow.python.keras.callbacks.History at 0x7f617c9fdb10>
'''
```

# Refferance
- [OpenCV를 이용한 MNIST 인식 모델 만들어보기](https://www.youtube.com/watch?v=TV3oplqa5VA)
- [Tensorflow를 이용한 CNN 실습영상](https://www.youtube.com/watch?v=pZGvMhhawy8)