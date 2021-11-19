---
title: '[Deep Learning] Hyperparameters'
description: 하이퍼파라미터 탐색 방법. 신경망에서 조정할 수 있는 주요 하이퍼 파라미터. Scikit-learn, Keras Tuner 등을 사용 구축한 신경망에 하이퍼파라미터 탐색 방법 적용
categories:
 - Deep Learning
tags: [Deep Learning, Hyperparameter, Grid Search, Random Search]
mathjax: enable
---

- [Hyperparameter Tuning Guide](https://www.youtube.com/watch?v=-i8b-srMhGM)
  - 딥러닝 홀로서기
- [Gradient Descent With Momentum](https://youtu.be/yWQZcdJ4k8s?t=34)
    - 학습해왔던 관성의 법칙을 유지하는 방식으로 학습 개선
- [Batch Size](https://youtu.be/U4WB9p6ODjM?t=29)
    - Batch를 크게하면 좋은 이유
    - 그러나 항상 크게할 수 없는 이유
    - 일반적으로 Batch라고 하면 Mini-batch를 의미한다는 점
- Wandb [QuickStart](https://docs.wandb.com/quickstart) 회원가입

## Review

- 신경망의 순전파와 역전파
    - 신경망의 순전파
    - 신경망의 역전파
    - 모델 생성과 모델 초기화
    - 경사하강법의 다양성
    - 학습 과정에서 알아야 할 Tricks
        - 가중치 감소/제한(Weight Decay/Constraint)
        - 드롭아웃(Dropout)
        - 학습률 계획(Learning Rate Scheduling)
- 그간 다뤄본 데이터
    - 손글씨 MNIST
    - Fashion MNIST

# Hyperparameter tuning

## Grid Search
- 오래걸리기 때문에 1개 혹은 2개 정도의 파라미터 최적값을 찾는 용도로 적합하다.
- 모델 성능에 직접적인 영향을 주는 하이퍼파라미터가 따로 있기 때문에 이러한 파라미터만 제대로 튜닝해서 최적값을 찾은 후 나머지 하이퍼파라미터도 조정해나가면 못해도 90% 이상의 성능을 확보할 수 있다.

## Random Search
- 무한루프라는 Grid Search의 단점을 해결하기 위해 나온 방법이다.
- 지정된 범위 내에서 무작위로 모델을 돌려본 후 최고 성능의 모델을 반환한다.
- 상대적으로 중요한 하이퍼파라미터에 대해서는 탐색을 더 하고, 덜 중요한 하이퍼파라미터에 대해서는 실험을 덜 하도록 한다.

## Bayesian Methods
- 이전 탐색 결과 정보를 새로운 탐색에 활용하는 방법이다.

## 튜닝 가능한 파라미터의 종류
- 배치 크기(batch_size)
- 반복 학습 횟수(에포크, training epochs)
- 옵티마이저(optimizer)
- 학습률(learning rate)
- 활성화 함수(activation functions)
- Regularization(weight decay, dropout 등)
- 은닉층(Hidden layer)의 노드(Node) 수

### Batch Size
- 모델의 가중치를 업데이트할 때마다, 즉 매 iteration 마다 몇 개의 입력 데이터를 볼 지 결정하는 하이퍼파라미터이다.
- 배치 사이즈를 너무 크게 하면 한 번에 많은 데이터에 대한 Loss를 계산해야 한다는 단점이 생긴다.
- 이럴 경우 가중치 업데이트가 빠르게 이루어지지 않은데다, 주어진 Epoch 안에 충분한 횟수의 iteration을 확보할 수 없게 된다.
- 파라미터가 굉장히 많은 모델에 큰 배치 사이즈를 적용하게 될 경우 메모리를 초과해버리는 현상(Out of Memory)이 발생하기도 한다.
- 배치사이즈를 너무 작게 하면 학습에 오랜 시간이 걸리고, 노이즈가 많이 발생한다는 단점이 있다.
- 일반적으로 35 ~ 512 사이의 2의 제곱수로 결정한다.
- 기본값은 32로 설정되어 있다.

### Optimizer
- adam이라는 옵티마이저가 꽤 좋은 성능을 보장한다.
- 기능을 추가한 adamW, adamP와 같은 옵티마이저도 사용된다.
- 중요한 점은 모든 경우에 좋은 옵티마이저란 없다는 것이다.

### Learning Rate

![image](https://user-images.githubusercontent.com/79494088/138241602-34f6bad6-13ab-498d-9fc5-4448925c52b5.png)

- 옵티마이저에서 지정해 줄 수 있는 하이퍼파라미터 중 하나이다.
- 학습률이 너무 높으면 경사 하강 과정에서 발산하면서 모델이 최적값을 찾을 수 없게 되어버린다.
- 반대로 너무 낮게 설정할 경우에는 최적점에 이르기까지 너무 오래 걸리거나, 주어진 iteration 내에서 모델이 수렴하는데 실패하기도 한다.

### Momentum

![image](https://user-images.githubusercontent.com/79494088/138241821-fef1e409-f188-446d-ac23-d71cad8f5bfb.png)

- 옵티마이저에 관성을 부여하는 하이퍼파라미터이다.
- 이전 iteration에서 경사 하강을 한 정도를 새로운 iteration에 반영한다.
- 지역 최적점에 빠지지 않도록 한다.

### Network Weight Initialization
- 초기 가중치를 어떻게 설정할 지 결정하는 가중치 초기화는 신경망에서 중요한 요소이다.

#### 표준편차 1인 정규분포로 가중치를 초기화할 때 각 층의 활성화 값 분포

![image](https://user-images.githubusercontent.com/79494088/138242085-6b5ad782-16fe-456c-bd1f-4c01ba621b20.png)

- 표준편차가 일정한 정규분포로 가중치를 초기화 해 줄 때 대부분 활성화 값이 0과 1에 위치한다.
- 활성값이 고르지 못할 경우 학습이 제대로 이루어지지 않는다.
- 간단한 방법임에도 잘 사용되지 않는다.

#### Xavier 초기화를 해주었을 때의 활성값 분포

![image](https://user-images.githubusercontent.com/79494088/138242423-ce27086c-7089-4d6d-a998-ee8b7ab1326b.png)

- Xavier initialization은 가중치를 표준편차가 고정값인 정규분포로 초기화 했을 때의 문제점을 해결하기 위해 등장한 방법이다.
- 이전 층의 노드가 $n$ 개일 때, 현재 층의 가중치를 표준편차가 $\frac{1}{\sqrt{n}}$ 인 정규분포로 초기화한다.

#### He 초기화를 해주었을 때 활성화 값 분포

![image](https://user-images.githubusercontent.com/79494088/138242761-561dab84-42a9-44a3-a032-55014e772a32.png)

- 활성화 함수가 sigmoid인 신경망에서는 잘 동작한다.
- 하지만 ReLU의 경우엔 층이 지날수록 활성값이 고르지 못하게 되는 문제를 보인다.
- 이런 문제를 해결하기 위해 등장한 것이 He initialization이다.
- He 초기화는 이전 층의 노드가 $n$ 개일 때, 현재 층의 가중치를 표준편차가 $\frac{2}{\sqrt{n}}$ 인 정규분포로 초기화한다.<br/>
- He 초기화를 적용하면 층이 지나도 활성값이 고르게 유지되는 것을 확인할 수 있다.

#### 요약
1. Sigmoid  ⇒  Xavier 초기화를 사용하는 것이 유리 
2. ReLU  ⇒  He 초기화 사용하는 것이 유리

# Code

## Import & EDA

```py
import pandas as pd
df = pd.read_csv('TelcomCustomer.csv')


# 중요한 TotalCharges를 numeric하게 변경
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')# coerce 설정 시 Number가 아닐 경우 NaN 설정

# 결측치 제거
df.dropna(axis=0, inplace=True)


# 원핫 인코딩
from category_encoders import OrdinalEncoder

encoder = OrdinalEncoder()
df_encoded = encoder.fit_transform(df)


# 타겟을 0과 1로 바꿔주기
df_encoded['Churn'] = df_encoded['Churn'].replace({1: 0, 2: 1})
df_encoded['Churn'].value_counts()
'''
0    5163
1    1869
Name: Churn, dtype: int64
'''


# 훈련, 테스트 셋을 나누기
from sklearn.model_selection import train_test_split

tr, t = train_test_split(df_encoded, test_size=0.25, random_state=1, stratify=df_encoded['Churn'])


# features 와 target 을 분리
target = 'Churn'
features = df_encoded.drop(columns=[target]).columns

xtr = tr[features]
xt = t[features]

ytr = tr[target]
yt = t[target]


# 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xtr = scaler.fit_transform(xtr)
xt = scaler.transform(xt)


# Baseline Modeling
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import keras
import tensorflow as tf
import IPython
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import kerastuner as kt


tf.random.set_seed(7)

model2 = Sequential()
model2.add(Dense(64, activation='relu'))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(1, activation='sigmoid')) # 이진분류이므로 노드수 1, 활성화 함수로는 시그모이드(sigmoid)

model2.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

results = model2.fit(xtr, ytr, epochs=10, validation_data=(xt, yt))
'''
Epoch 1/10
165/165 [==============================] - 1s 3ms/step - loss: 0.4604 - accuracy: 0.7710 - val_loss: 0.4491 - val_accuracy: 0.7912
Epoch 2/10
165/165 [==============================] - 0s 2ms/step - loss: 0.4164 - accuracy: 0.8019 - val_loss: 0.4461 - val_accuracy: 0.7986
Epoch 3/10
165/165 [==============================] - 0s 2ms/step - loss: 0.4071 - accuracy: 0.8066 - val_loss: 0.4482 - val_accuracy: 0.7918
Epoch 4/10
165/165 [==============================] - 0s 2ms/step - loss: 0.4018 - accuracy: 0.8117 - val_loss: 0.4452 - val_accuracy: 0.8020
Epoch 5/10
165/165 [==============================] - 0s 2ms/step - loss: 0.3969 - accuracy: 0.8159 - val_loss: 0.4512 - val_accuracy: 0.7947
Epoch 6/10
165/165 [==============================] - 0s 2ms/step - loss: 0.3943 - accuracy: 0.8151 - val_loss: 0.4527 - val_accuracy: 0.7901
Epoch 7/10
165/165 [==============================] - 0s 2ms/step - loss: 0.3900 - accuracy: 0.8176 - val_loss: 0.4520 - val_accuracy: 0.7958
Epoch 8/10
165/165 [==============================] - 0s 2ms/step - loss: 0.3886 - accuracy: 0.8180 - val_loss: 0.4533 - val_accuracy: 0.7929
Epoch 9/10
165/165 [==============================] - 0s 2ms/step - loss: 0.3836 - accuracy: 0.8210 - val_loss: 0.4551 - val_accuracy: 0.7856
Epoch 10/10
165/165 [==============================] - 0s 2ms/step - loss: 0.3799 - accuracy: 0.8225 - val_loss: 0.4638 - val_accuracy: 0.7810
'''


# 테스트셋 사용해서 결과 보기
model2.evaluate(xt,  yt, verbose=2) 
'''
55/55 - 0s - loss: 0.4638 - accuracy: 0.7810
[0.4637787342071533, 0.7810011506080627]
'''
```

## GridSearchCV 사용

```py
# 모델 만들기
tf.random.set_seed(7)

def model_builder(nodes=16, activation='relu'):

  model = Sequential()
  model.add(Dense(nodes, activation=activation))
  model.add(Dense(nodes, activation=activation))
  model.add(Dense(1, activation='sigmoid')) # 이진분류니까 노드수 1, 활성함수로는 시그모이드

  model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])

  return model

# keras.wrapper를 활용하여 분류기
model = KerasClassifier(build_fn=model_builder, verbose=0)

# GridSearch
batch_size = [64, 128, 256]
epochs = [10, 20, 30]
nodes = [64, 128, 256]
activation = ['relu', 'sigmoid']
param_grid = dict(batch_size=batch_size, epochs=epochs, nodes=nodes, activation=activation)


# GridSearch CV를 만들기
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
grid_result = grid.fit(xtr, ytr)
'''
Fitting 3 folds for each of 54 candidates, totalling 162 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed:  1.5min
[Parallel(n_jobs=-1)]: Done 162 out of 162 | elapsed:  4.5min finished
'''


# 최적의 결과값을 낸 파라미터를 출력
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"Means: {mean}, Stdev: {stdev} with: {param}")
'''
Best: 0.8060295979181925 using {'activation': 'sigmoid', 'batch_size': 64, 'epochs': 20, 'nodes': 256}
Means: 0.8031854232152303, Stdev: 0.0040489550995696615 with: {'activation': 'relu', 'batch_size': 64, 'epochs': 10, 'nodes': 64}
Means: 0.7957906723022461, Stdev: 0.0012288098144601328 with: {'activation': 'relu', 'batch_size': 64, 'epochs': 10, 'nodes': 128}
Means: 0.7912400563557943, Stdev: 0.0021283822713404434 with: {'activation': 'relu', 'batch_size': 64, 'epochs': 10, 'nodes': 256}
'
'
'
Means: 0.8010997176170349, Stdev: 0.0011688358813985366 with: {'activation': 'sigmoid', 'batch_size': 256, 'epochs': 30, 'nodes': 64}
Means: 0.8024269938468933, Stdev: 0.003295064887139762 with: {'activation': 'sigmoid', 'batch_size': 256, 'epochs': 30, 'nodes': 128}
Means: 0.8022373914718628, Stdev: 0.005094801996467334 with: {'activation': 'sigmoid', 'batch_size': 256, 'epochs': 30, 'nodes': 256}
'''
```

## Keras Tuner

```py
# 모델 만들기
def model_builder(hp):

  model = Sequential()

  # Dense layer에서 노드 수를 조정(32-512)
  hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)

  model.add(Dense(units = hp_units, activation='relu'))
  model.add(Dense(units = hp_units, activation='relu'))

  model.add(Dense(1, activation='sigmoid')) # 이진분류니까 노드수 1, 활성함수로는 시그모이드

  # Optimizer의 학습률(learning rate)을 조정[0.01, 0.001, 0.0001]합니다. 
  hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

  # 컴파일 단계, 옵티마이저와 손실함수, 측정지표를 연결해서 계산 그래프를 구성함
  model.compile(optimizer=keras.optimizers.Adam(learning_rate = hp_learning_rate), 
                loss=keras.losses.BinaryCrossentropy(from_logits = True), 
                metrics=['accuracy'])

  return model


# 튜너를 인스턴스화하고 하이퍼 튜닝을 수행
from tensorflow import keras

tuner = kt.Hyperband(model_builder,
                     objective = 'val_accuracy', 
                     max_epochs = 30, 
                     factor = 3,
                     directory = 'my_dir',
                     project_name = 'intro_to_kt')


# callback 정의 : 하이퍼 파라미터 검색을 실행하기 전에 모든 교육 단계가 끝날 때마다 교육 출력을 지우도록 콜백을 정의합니다.

class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)   


tuner.search(xtr, ytr, epochs = 30, batch_size=50, validation_data = (xt, yt), callbacks = [ClearTrainingOutput()])

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
최적화된 Dense 노드 수 : {best_hps.get('units')} 
최적화된 Learning Rate : {best_hps.get('learning_rate')} 
""")
'''
Trial 66 Complete [00h 00m 02s]
val_accuracy: 0.7997724413871765

Best val_accuracy So Far: 0.8037542700767517
Total elapsed time: 00h 02m 53s
INFO:tensorflow:Oracle triggered exit

최적화된 Dense 노드 수 : 384 
최적화된 Learning Rate : 0.0001 
'''


from tensorflow.keras import regularizers

tf.random.set_seed(1442)
initializer = tf.keras.initializers.HeNormal()

model = Sequential()

model.add(Dense(best_hps.get('units'), 
                activation='relu', kernel_initializer=initializer,          
                kernel_regularizer=regularizers.l2(0.01),    # L2 norm regularization
                activity_regularizer=regularizers.l1(0.01))) # L1 norm regularization))
model.add(Dense(best_hps.get('units'),
                activation='relu', kernel_initializer=initializer,            
                kernel_regularizer=regularizers.l2(0.01),    # L2 norm regularization
                activity_regularizer=regularizers.l1(0.01)))
model.add(Dense(1, activation='sigmoid')) # 이진분류니까 노드수 1, 활성함수로는 시그모이드

model.compile(optimizer=keras.optimizers.Adam(learning_rate = best_hps.get('learning_rate')), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

results = model.fit(xtr, ytr, epochs=22, batch_size=50, validation_data=(xt, yt))
'''
Epoch 1/22
106/106 [==============================] - 1s 8ms/step - loss: 18.8031 - accuracy: 0.7455 - val_loss: 17.5152 - val_accuracy: 0.7668
Epoch 2/22
106/106 [==============================] - 1s 6ms/step - loss: 16.6125 - accuracy: 0.7799 - val_loss: 15.8043 - val_accuracy: 0.7810
Epoch 3/22
106/106 [==============================] - 1s 6ms/step - loss: 15.0911 - accuracy: 0.7907 - val_loss: 14.4226 - val_accuracy: 0.7890
'
'
'
Epoch 20/22
106/106 [==============================] - 1s 5ms/step - loss: 4.3125 - accuracy: 0.8077 - val_loss: 4.2100 - val_accuracy: 0.7992
Epoch 21/22
106/106 [==============================] - 1s 6ms/step - loss: 4.0653 - accuracy: 0.8093 - val_loss: 3.9708 - val_accuracy: 0.8038
Epoch 22/22
106/106 [==============================] - 1s 6ms/step - loss: 3.8348 - accuracy: 0.8083 - val_loss: 3.7482 - val_accuracy: 0.8060
'''


# 테스트셋 사용해서 결과 보기
model.evaluate(xt,  yt, verbose=2)
'''
55/55 - 0s - loss: 3.7482 - accuracy: 0.8060
[3.748224973678589, 0.8060295581817627]
'''
```

# Refferance
- [Grid Search Hyperparameters for Deep Learning](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
- [Hyperparameters Optimization for Deep Learning Models](https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/)
- [Dropout Regularization in Deep Learning](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)
- [Weight Constraints in Deep Learning](https://machinelearningmastery.com/introduction-to-weight-constraints-to-reduce-generalization-error-in-deep-learning/)
- [Number of Layers and Nodes in a Neural Network](https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/)
- [Batch Normalization](https://shuuki4.wordpress.com/2016/01/13/batch-normalization-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84/)