---
title: '[DL mini Project] LSTM 애플(APPL) 주가 예측'
description: LSTM을 이용하여 애플 주식을 예측하는 모델 구현
categories:
 - Project
tags: [DL, project, 애플, APPL, 주가 예측, 주식]
mathjax: enable
---

# 개요
- 오늘은 LSTM을 배운 날!
- LSTM을 이용해 Deep Learning 주가 예측 머신을 만드는 튜토리얼이 많아 따라해보며 실습해보기로 한다.
- 나름 얼마 전 테슬라로 재미 좀 보고, 애플을 사랑하는 가치 투자 주주로써 애플을 활용하여 주가 예측을 해 볼 예정이다.

# EDA & 전처리 & FE
- 주가 데이터는 야후 파이넨셜을 이용하여 다운로드했다.

```py
import pandas as pd

df = pd.read_csv('AAPL.csv')
df
```

![image](https://user-images.githubusercontent.com/79494088/139005459-099b8b54-efb2-42e1-90ea-1798e1b63e37.png)

```py
# datetime 변경 및 날짜 분할
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['Year'] =df['Date'].dt.year
df['Month'] =df['Date'].dt.month
df['Day'] =df['Date'].dt.day
df
```

![image](https://user-images.githubusercontent.com/79494088/139005548-afbc4c77-c470-4cb7-9f55-24247a9ef3d4.png)

### Visualization

```py
# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 9))
sns.lineplot(y=df['Close'], x=df['Date'])
plt.xlabel('time')
plt.ylabel('price')
```

![image](https://user-images.githubusercontent.com/79494088/139005618-522df46c-3747-4e60-a77e-7cf493e0ad6d.png)

### Nomalization

```py
# 정규화
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
df_scaled = scaler.fit_transform(df[scale_cols])

df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

df_scaled
```

### 학습 데이터셋 생성
- `window_size`: 얼마동안의 주가 데이터에 기반하여 다음날 종가를 예측할 것인가를 정하는 parameter로써, 과거 20일을 기반으로 내일 데이터를 예측한다고 가정하면 window_size=20이 된다.
- `test_size`: 과거부터 200일 이전의 데이터를 학습하게 되고, test를 위해 이후 200일의 데이터로 모델이 주가를 예측하도록 한 다음, 실제 데이터와 오차가 얼마나 있는지 확인한다.

```py
import numpy as np

TEST_SIZE = 200

train = df_scaled[:-TEST_SIZE]
test = df_scaled[-TEST_SIZE:]

# 순차적으로 20일 동안의 데이터셋을 묶고, label과 함께 return 해주는 함수 생성
def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)
```

### Feature Label 정의

```py
feature_cols = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
label_cols = ['Close']

train_feature = train[feature_cols]
train_label = train[label_cols]

# train dataset
train_feature, train_label = make_dataset(train_feature, train_label, 20)

# train, validation set 생성
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

x_train.shape, x_valid.shape
'''
((7890, 20, 5), (1973, 20, 5))
'''


# test dataset (실제 예측 해볼 데이터)
test_feature = test[feature_cols]
test_label = test[label_cols]

test_feature, test_label = make_dataset(test_feature, test_label, 20)
test_feature.shape, test_label.shape
'''
((180, 20, 5), (180, 1))
'''
```

# Keras 활용 LSTM 모델 생성

```py
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(16, 
               input_shape=(train_feature.shape[1], train_feature.shape[2]), 
               activation='relu', 
               return_sequences=False)
          )
model.add(Dense(1))


model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)


filename = 'tmp_checkpoint.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_train, y_train, 
                    epochs=200, 
                    batch_size=16,
                    validation_data=(x_valid, y_valid), 
                    callbacks=[early_stop, checkpoint])
'''
Epoch 1/200
494/494 [==============================] - 7s 11ms/step - loss: 3.1618e-04 - val_loss: 6.6918e-05

Epoch 00001: val_loss improved from inf to 0.00007, saving model to tmp_checkpoint.h5
Epoch 2/200
494/494 [==============================] - 5s 10ms/step - loss: 3.5111e-05 - val_loss: 4.9173e-05
'
'
'
Epoch 00021: val_loss did not improve from 0.00001
Epoch 22/200
494/494 [==============================] - 5s 10ms/step - loss: 1.4883e-05 - val_loss: 1.7614e-05

Epoch 00022: val_loss did not improve from 0.00001
Epoch 23/200
494/494 [==============================] - 5s 10ms/step - loss: 1.5925e-05 - val_loss: 2.1189e-05

Epoch 00023: val_loss did not improve from 0.00001
'''


# weight 로딩
model.load_weights(filename)

# 예측
pred = model.predict(test_feature)
```

# 실제 & 예측 시각화

```py
plt.figure(figsize=(12, 9))
plt.plot(test_label, label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/139007311-60ed03c3-6709-49a2-98e1-f582ed5cf617.png)

# 결론
- 시각화 결과를 보고 그럴싸해 보이지만, 많은 허점이 있다.
- [팩폭을 맞은 블로그 포스팅 바로가기](https://codingapple.com/unit/deep-learning-stock-price-ai/)
- 간단하게 요약해보자면, 위 모델은 그저 예측 주가가 그 전날의 주가를 그대로 표시한다는 것이다.
- 컴퓨터에게 '28일의 주식가격'을 물어보면 '27일의 주식가격'이라고 답변하는 것과 같다.
- 모델이 loss값이 최소화하려고 할때 가장 적은 모델이 그저 '내일 주식가격은 오늘 주식가격과 똑같을 것이다'가 된다.
- 잘못된 건 아니지만, 쓸모가 전혀 없다.

## 해결책
- 전날 가격보다, 외부 정보가 중요할 수 있다.
  - 전날 거래량
  - SNS 언급량
  - 전날 나스닥지수 증감량
  - 관련 업종 주가추이

## 생각
- 예쁘고 쓸모 없는 모델을 만들었다.
- 하지만 LSTM을 이해하는데 큰 도움이 되었다고 생각한다(재미와 더불어).
- 주가 예측은 현실적으로 어려우니(현실적으로 가능했으면 딥러닝을 배운사람들은 때부자가 됐겠다.) 일획천금 노릴 생각하지말고 열심히 살자.

# Reference
- [딥러닝(LSTM)을 활용하여 삼성전자 주가 예측을 해보았습니다](https://teddylee777.github.io/tensorflow/LSTM%EC%9C%BC%EB%A1%9C-%EC%98%88%EC%B8%A1%ED%95%B4%EB%B3%B4%EB%8A%94-%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90-%EC%A3%BC%EA%B0%80)
- [주식 주가 데이터 다운로드 방법 정리](https://muzukphysics.tistory.com/entry/%EC%A3%BC%EC%8B%9D-%EC%A3%BC%EA%B0%80-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C-%EB%B0%A9%EB%B2%95-%EC%A0%95%EB%A6%AC)
- [(칼럼) 딥러닝 초보들이 흔히하는 실수 : 주식가격 예측 AI](https://codingapple.com/unit/deep-learning-stock-price-ai/)