---
title: '[Deep Learning] Autoencoder'
description: 오토인코더의 구성과 학습과정, Latent variable 추상적 개념 정의, 기본적인 information retrieval problem 해결, AE 활용방안
categories:
 - Deep Learning
tags: [Deep Learning, Autoencoder, AE, Latent variable, information retrieval problem]
mathjax: enable
---

# Warm Up
- [AE 소개영상 1](https://youtu.be/YxtzQbe2UaE)
- [AE 소개영상 2](https://youtu.be/54hyK1J4wTc?t=889)
- [AE 소개영상 3(영어)](https://youtu.be/3jmcHZq3A5s?t=93)

# Autoencoders
- 입력데이터 자체를 레이블로 활용하는 학습방식이다.
- 별도의 레이블이 필요하지 않은 비지도 학습 방식이다.
- 데이터 코딩(encoding, decoding)을 위해서 원하는 차원만 할당하면 자동으로 학습하여 원하는 형태로 데이터의 차원을 축소해주는 신경망의 한 어플리케이션이다.
- 오토 인코더의 목적은 네트워크가 중요한 의미를 갖는 신호 외의 노이즈를 제거하도록 훈련함으로써 일반적인 차원 축소 방법론과 비슷한 목적으로 활용된다.
- 여기서 코딩된 코드 영역을 입력 데이터의 잠재적 표현(Latent representation)이라고 부르게 된다.
- 데이터를 잠재적 표현으로 변경하는 차원축소과정(encoding)과 잠재적 표현에서 다시 데이터로 재구성과정(decoding)은 각각의 가중치의 연산을 통해 이뤄지고, 이를 신경망을 통해 학습한다.
- 학습 과정에서는 인코딩 모델(파라미터)와 디코딩 모델(파라미터)가 동시에 학습되지만, 이를 각각의 모델로 구분하여 사용할 수 있다.
- 오토 인코더를 통하여 데이터의 노이즈를 제거하도록 학습하면 노이즈가 제거된 데이터의 특징값을 추출하는 특성값 추출기(Feature extractor)로 활용할 수 있고, 이 모델을 다양한 방식으로 활용할 수 있다.
- label 없이도 이미지를 일정 크기의 벡터로 자동 표현 할 수 있을까?
- 고정 크기의 벡터로 표현해야 하는 이유
    - __Information Retrieval__
        - [Reverse Image Search](https://en.wikipedia.org/wiki/Reverse_image_search)
        - [Recommendation Systems - Content Based Filtering](https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering)
    - __Dimensionality Reduction__
        - [Feature Extraction](https://www.kaggle.com/c/vsb-power-line-fault-detection/discussion/78285)
        - [Manifold Learning](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction)
- 오토인코더를 사용하여 Word2Vec처럼, Image2Vec형태로 이미지를 벡터화 하는 것과 유사한 목표를 달성할 수 있다.
- 오토인코더는 '출력 = 입력'을 학습시키는 신경망이다.
- '비지도학습'이라고도 하지만, Self-supervised learning(SSL)의 워딩과 비슷한 개념이다.
- 학습이 완료된 후에도 일반적으로 입력과 완벽히 똑같은 출력을 만들 수 없겠지만, 매우 근사한 수치로 복제된다.
- Loss func에서 입력과 출력의 차이를 이용하기 때문에 출력은 입력과 최대한 근사하게 만들기 위해서 Latent representation으로 encoding하고, 다시 data로 decoding 하면서 이 latent는 데이터의 정보량이 많은 부분, 또는 대표적인 속성을 우선 학습하게 된다.
- 이렇기 때문에 AE를 생성 모델적 접근 방식에서도 중요한 부분으로 다뤄지게 된다.
- 생성모델의 대표적인 생성적 대립 신경망(GAN)에서도 배우지만, 인코더와 디코더가 있는 구조를 이용해 생성모델을 구현할 수 있다.
- 오토 인코더는 다양한 방식으로 활용할 수 있지만, 기존 신경망의 원리가 동일하며, 또한 신경망의 특이 케이스라고 생각하면 된다.
- 기존에 배운 역전파 및 경사하강법은 여기에도 잘 동작한다.

<img src='https://miro.medium.com/max/1400/1*44eDEuZBEsmG_TCAKRI3Kw@2x.png'>

- Input에서 Code(Latent)까지 `Encoder`는 입력 데이터를 압축하고 `Decoder`는 압축되지 않은 버전의 데이터를 생성하여 가능한 정확하게 입력을 재구성한다.
- 학습과정(minimizing a loss function): $ L(x, g(f(x))) $
- $L$ 은 손실함수, $g(f(x))$와 $x$의 dissimiliarity (예, mean squared error)
- $f$ 는 encoder function
- $g$ 는 decoder function

## Stacked AE

![image](https://user-images.githubusercontent.com/79494088/139848824-bda0615f-db80-40ec-8478-d34da656ada6.png)

<img src="https://kr.mathworks.com/help/deeplearning/ug/autoencoderdigitsexample_06_ko_KR.png"/>

<img src="https://kr.mathworks.com/help/deeplearning/ug/autoencoderdigitsexample_07_ko_KR.png"/>

<img src="https://kr.mathworks.com/help/deeplearning/ug/autoencoderdigitsexample_08_ko_KR.png"/>

- 여러개의 히든 레이어를 가지는 오토인코더이며, 레이어를 추가할수록 오토인코더가 더 복잡한 코딩(부호화)을 학습할 수 있다.
- 히든레이어를 기준으로 대칭인 구조를 가진다.

# 실습 1

```py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
```

## 기본 오토 인코더

![image](https://user-images.githubusercontent.com/79494088/139857262-6d65f8a0-f237-4b00-a7c5-5052a51e7f12.png)

- 이미지를 64차원 잠재 벡터로 압축하는 가장 간단한 형태의 encoder와 잠재 공간에서 원본 이미지를 재구성하는 decoder라는 두 개의 Dense 레이어로 구성된 모델이다.
- 모델 정의: [Keras Model Subclassing API](https://www.tensorflow.org/guide/keras/custom_layers_and_models)

```py
# 데이터셋
(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)
'''
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
32768/29515 [=================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26427392/26421880 [==============================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
8192/5148 [===============================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4423680/4422102 [==============================] - 0s 0us/step
(60000, 28, 28)
(10000, 28, 28)
'''


# Code영역의 벡터(= Latent vector)의 수 정의
latent_dim = 64 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='sigmoid'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
# decoded된 데이터를 output으로 설정함
autoencoder = Autoencoder(latent_dim) 


autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())


# x_train을 입력 및 대상으로 사용하여 모델을 훈련시킨다.
# encoder는 데이터 세트를 784차원에서 latent 공간으로 압축하는 방법을 배우고 decoder는 원본 이미지를 재구성하는 방법을 배운다.
autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))
'''
Epoch 1/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0238 - val_loss: 0.0138
Epoch 2/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0117 - val_loss: 0.0107
Epoch 3/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0100 - val_loss: 0.0097
Epoch 4/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0094 - val_loss: 0.0095
Epoch 5/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0091 - val_loss: 0.0091
Epoch 6/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0090 - val_loss: 0.0090
Epoch 7/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0089 - val_loss: 0.0090
Epoch 8/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0088 - val_loss: 0.0089
Epoch 9/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0088 - val_loss: 0.0088
Epoch 10/10
1875/1875 [==============================] - 3s 2ms/step - loss: 0.0087 - val_loss: 0.0088
<tensorflow.python.keras.callbacks.History at 0x7f380032b588>
'''


# 학습된 모델 이용 테스트셋의 이미지를 인코딩 및 디코딩하여 모델을 테스트한다.
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/139858031-06e6ada2-90bc-4480-9c5b-5f5ff7343fb9.png)

# 실습 2

## 노이즈 제거용 오토 인코더

![image](https://user-images.githubusercontent.com/79494088/139858208-448b5305-66b8-4ef1-96d5-bdd9316db758.png)

- 오토인코더는 이미지에서 노이즈를 제거하도록 훈련될 수도 있다.
- 각 이미지에 임의의 노이즈를 적용하여 Fashion MNIST 데이터셋의 노이즈 버전을 생성한다.
- 다음 잡음이 있는 이미지를 입력으로 사용하고 원본 이미지를 대상으로 사용하여 오토 인코더를 훈련한다.

```py
(x_train, _), (x_test, _) = fashion_mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print(x_train.shape)
'''
(60000, 28, 28, 1)
'''


# 이미지에 random noise를 만들어 더한다.
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape) 

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)


# 노이즈가 더해진 이미지를 시각회
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/139859447-1505a696-3302-4c91-9c34-bb7eed6e1803.png)

## Convolutional autoencoder
- 가중치의 형태를 CNN으로 가져온다.
- [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) 레이어를 `encoder`로 사용하고, 반대로 `decoder`로는 [Conv2DTranspose](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose)를 사용하는 구조이다.

```py
# CNN의 형태를 갖는 autoencoder 코드
class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(28, 28, 1)), 
      layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2)])
    
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Denoise()


autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())


autoencoder.fit(x_train_noisy, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))
'''
Epoch 1/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0175 - val_loss: 0.0107
Epoch 2/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0098 - val_loss: 0.0091
Epoch 3/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0088 - val_loss: 0.0086
Epoch 4/10
1875/1875 [==============================] - 5s 2ms/step - loss: 0.0084 - val_loss: 0.0083
Epoch 5/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0081 - val_loss: 0.0081
Epoch 6/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0080 - val_loss: 0.0080
Epoch 7/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0079 - val_loss: 0.0079
Epoch 8/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0078 - val_loss: 0.0078
Epoch 9/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0077 - val_loss: 0.0076
Epoch 10/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0076 - val_loss: 0.0076
<tensorflow.python.keras.callbacks.History at 0x7f37ac027b38>
'''


# 이미지가 어떻게 다운 샘플링되는지 확인
autoencoder.encoder.summary()
'''
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 14, 14, 16)        160       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 8)           1160      
=================================================================
Total params: 1,320
Trainable params: 1,320
Non-trainable params: 0
_________________________________________________________________
'''


# 디코더 구조
autoencoder.decoder.summary()
'''
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_transpose (Conv2DTran (None, 14, 14, 8)         584       
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 28, 28, 16)        1168      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 1)         145       
=================================================================
Total params: 1,897
Trainable params: 1,897
Non-trainable params: 0
_________________________________________________________________
'''


# 이미지 변환 확인
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/139860008-b0031660-ff77-47b0-825b-d33d781df802.png)

# 실습 3

## 이상현상 발견 용 오토 인코더
- [ECG 5000 데이터 세트](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000)에서 이상현상을 감지하도록 오토인코더를 훈련시킨다.
- 각각 140개의 데이터로 구성된 5000개의 [심전도(ECG, Electrocardiograms)](https://en.wikipedia.org/wiki/Electrocardiography)가 포함되어 있다.
- 각 예제는 0(비정상 리듬) 또는 1(정상 리듬)으로 레이블이 지정된 단순화 된 버전의 데이터셋을 사용한다.
- 이 예제는 레이블이 지정된 데이터셋이므로 지도학습 문제라고 할 수 있다.
- 이 예제의 목표는 사용 가능한 레이블이 없는 더 큰 데이터셋에 적용할 수 있는 이상현상 감지 개념을 설명하는 것이다.
- 오토 인코더는 재구성 오류를 최소화하도록 훈련되었다.
- 오토 인코더를 정상적인 리듬으로만 훈련한 뒤 이를 사용하여 모든 데이터를 재구성한다.
- 여기서 우리의 가설은 비정상적인 리듬이 더 높은 재건(reconstruction) 오류를 가질 것이라는 것이다.
- 이것을 이용하여 오류가 임계값을 초과하는 경우 리듬을 이상으로 분류하는 것이다.

```py
# 데이터셋을 불러옵니다. 
dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
dataframe.head()
```

![image](https://user-images.githubusercontent.com/79494088/139862782-d85fe4d6-fa61-40a1-b48a-4721fa3d2ace.png)

```py
# 마지막으로 포함된 레이블을 따로 저장
labels = raw_data[:, -1]

# 데이터에서는 레이블을 제거
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)


# min-max 알고리즘을 이용하여 정규화합니다. 
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)


# 정상 리듬만 사용하여 오토 인코더 훈련
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]


# ECG 파장 플로팅
plt.grid()
plt.plot(np.arange(140), normal_train_data[0])
plt.title("A Normal ECG")
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/139863036-a0342b71-590e-4e49-bbe7-422950cbbc86.png)

```py
# ECG 데이터 역시 플로팅
plt.grid()
plt.plot(np.arange(140), anomalous_train_data[0])
plt.title("An Anomalous ECG")
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/139863142-53867b73-b45e-4eb3-baf6-52d3f90ad934.png)

```py
# 학습 모델 구축
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()


autoencoder.compile(optimizer='adam', loss='mae')


# 정상 ECG만 사용하고 훈련하고, 테스트 용도에서는 정상과 비정상이 섞인 상태를 평가한다.
history = autoencoder.fit(normal_train_data, normal_train_data, 
          epochs=20, 
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)
'''
Epoch 1/20
5/5 [==============================] - 0s 21ms/step - loss: 0.0581 - val_loss: 0.0534
Epoch 2/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0557 - val_loss: 0.0514
Epoch 3/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0526 - val_loss: 0.0498
Epoch 4/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0491 - val_loss: 0.0471
Epoch 5/20
5/5 [==============================] - 0s 6ms/step - loss: 0.0454 - val_loss: 0.0453
Epoch 6/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0419 - val_loss: 0.0436
Epoch 7/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0384 - val_loss: 0.0418
Epoch 8/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0353 - val_loss: 0.0407
Epoch 9/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0325 - val_loss: 0.0391
Epoch 10/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0302 - val_loss: 0.0380
Epoch 11/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0285 - val_loss: 0.0372
Epoch 12/20
5/5 [==============================] - 0s 6ms/step - loss: 0.0272 - val_loss: 0.0365
Epoch 13/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0260 - val_loss: 0.0356
Epoch 14/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0251 - val_loss: 0.0349
Epoch 15/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0242 - val_loss: 0.0345
Epoch 16/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0235 - val_loss: 0.0338
Epoch 17/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0229 - val_loss: 0.0331
Epoch 18/20
5/5 [==============================] - 0s 6ms/step - loss: 0.0222 - val_loss: 0.0327
Epoch 19/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0216 - val_loss: 0.0322
Epoch 20/20
5/5 [==============================] - 0s 5ms/step - loss: 0.0211 - val_loss: 0.0318
'''


plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
```

![image](https://user-images.githubusercontent.com/79494088/139863532-f5c3caca-e44f-4885-bf52-8fd4b73c7634.png)

- 재구성 오류가 일반 ECG의 표준 편차 1보다 큰 경우 ECG를 비정상으로 분류하도록 정의한다.
- 학습 세트에서 일반 ECG, 오토 인코더에 의해 인코딩 및 디코딩 된 후의 재구성 및 재구성 오류를 플로팅 한다.

```py
encoded_imgs = autoencoder.encoder(normal_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

plt.plot(normal_test_data[0],'b')
plt.plot(decoded_imgs[0],'r')
plt.fill_between(np.arange(140), decoded_imgs[0], normal_test_data[0], color='lightcoral' )
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/139863752-0d1ba6b7-c6c8-4e3a-a7b0-638da046e98e.png)

```py
# 같은 방식의 이상현상 샘플 플로팅
encoded_imgs = autoencoder.encoder(anomalous_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

plt.plot(anomalous_test_data[0],'b')
plt.plot(decoded_imgs[0],'r')
plt.fill_between(np.arange(140), decoded_imgs[0], anomalous_test_data[0], color='lightcoral' )
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/139863861-3459e900-0dd5-4f2b-888d-c048d650bb91.png)

## 이상현상 탐지
- 재구성 손실이 고정 임계 값보다 큰지 여부를 계산하여 이상을 감지한다.
- 이 튜토리얼에서는 훈련 세트의 정상 예제에 대한 MAE를 계산한 다음 재구성 오차가 훈련 세트의 표준 편차보다 큰 경우 향우 예제를 비정상적인 것으로 분류한다.

```py
# 훈련 세트의 정상 ECG에 대한 재구성 오류 플로팅
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

plt.hist(train_loss, bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/139864177-35cb0ef0-8919-4ae0-bad6-8475248b9e64.png)

- 평균보다 1SD가 높은 임계값을 선택한다.
- 통계수준에서 2SD를 벗어나게 되면, 95%의 신뢰구간을 벗어나는 것을 의미한다.

```py
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)
'''
Threshold:  0.032173388
'''

reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

plt.hist(test_loss, bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/139864887-8c871907-ace9-4382-8b6c-8f02794892f3.png)

```py
# 재구성 오류가 임계값보다 큰 경우 심전도를 이상 현상으로 분류하도록 설정
def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, preds)))
  print("Precision = {}".format(precision_score(labels, preds)))
  print("Recall = {}".format(recall_score(labels, preds)))


preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)
'''
Accuracy = 0.943
Precision = 0.9921722113502935
Recall = 0.9053571428571429
'''
```

# Review

## 오토인코더(AutoEncoder, AE)

![image](https://user-images.githubusercontent.com/79494088/140001296-7abfd426-866f-47de-92be-f66187ec0184.png)

- 단순히 입력을 출력으로 복사하는 신경망이다.
- 간단한 신경망처럼 보이지만 네트워크에 여러가지 방법으로 제약을 줌으로써 어려운 신경망을 만든다.
- hidden layer의 뉴런 수를 input layer 보다 작게 해서 데이터를 압축(차원을 축소)한다거나, 입력 데이터에 노이즈(noise)를 추가한 후 원본 입력을 복원할 수 있도록 네트워크를 학습시키는 등 다양한 오토인코더가 있다.
- 이러한 제약들은 오토인코더가 단순히 입력을 바로 출력으로 복사하지 못하도록 방지하며, 데이터를 효율적으로 표현(representataion)하는 방법을 학습하도록 제어한다.

### 잠재 벡터(Latent Vector)
- 차원이 줄어든 채로 데이터를 잘 설명할 수 있는 잠재 공간에서의 벡터

### 이상치 탐지(Anomaly Detection)
- Anomaly Detection이란, Normal(정상) sample과 Abnormal(비정상, 이상치, 특이치) sample을 구별해내는 문제를 의미한다.
- Anomaly Detection은 학습 데이터 셋에 비정상적인 sample이 포함되는지, 각 sample의 label이 존재하는지, 비정상적인 sample의 성격이 정상 sample과 어떻게 다른지, 정상 sample의 class가 단일 class 인지 Multi-class 인지 등에 따라 다른 용어를 사용한다.

### 매니폴드(Manifold), 혹은 매니폴드 학습(Manifold Learning)
- Manifold란 고차원 데이터(e.g Image의 경우 (256, 256, 3) or...)가 있을 때 고차원 데이터를 데이터 공간에 뿌리면 sample들을 잘 아우르는 subspace가 있을 것이라는 가정에서 학습을 진행하는 방법이다.
- 이렇게 찾은 manifold는 데이터의 차원을 축소시킬 수 있다.


## Stacked AutoEncoder(Stacked AE)

![image](https://user-images.githubusercontent.com/79494088/140001515-a6fabea0-076d-4cbb-b905-c08fdf49f4f9.png)

- Stacked 오토 인코더는 여러개의 히든 레이어를 가지는 오토인코더이며, 레이어를 추가할수록 오토인코더가 더 복잡한 코딩을 할 수 있다.
- 히든 레이어를 기준으로 대칭인 구조를 가진다.

## Denoising AutoEncoder (DAE)

![image](https://user-images.githubusercontent.com/79494088/140001670-4f5953b1-76da-4c08-9665-1b28837cef8e.png)

- 오토인코더가 의미있는 특성을 학습하도록 제약을 주는 다른 방법은 입력에 noise를 추가하고, 노이즈가 없는 원본 입력을 재구성하여 학습시키는 것이다.
- 노이즈는 Gaussian 노이즈를 추가하거나, dropout처럼 랜덤하게 입력 노드를 꺼서 발생 시킬 수 있다.

## Variational AutoEncoder (VAE)

![image](https://user-images.githubusercontent.com/79494088/140001941-ef6d26b1-8003-40ba-a853-b81a855f1b9c.png)

- 2014년 D.Kingma와 M.Welling이 [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114v10.pdf) 논문에서 제안한 오토인코더의 한 종류이다.
- 확률적 오토인코더(probabilistic autoencoder)로 학습이 끝난 후에도 출력이 부분적으로 우연에 의해 결정된다.
- 생성 오토인코더(generatie autoencoder)로, 학습 데이터셋에서 샘플링된 것과 같은 새로운 샘플을 생성할 수 있다.
- VAE의 코딩층은 다른 오토인코더와 다른 부분이 있는데 주어진 입력에 대해 바로 코딩을 만드는 것이 아니라, 인코더는 평균 코딩과 표준 편차 코딩을 만든다.
- 실제 코딩은 가우시안 분포에서 랜덤하게 샘플링되며, 이렇게 샘플링 된 코딩을 디코더가 원본 입력으로 재구성하게 된다.
- VAE는 마치 가우시안 분포에서 샘플링된 것처럼 보이는 코딩을 만드는 경향이 있는데, 학습하는 동안 손실함수가 코딩을 가우시안 샘플의 집합처럼 보이는 형태를 가진 코딩 공간 또는 잠재변수공간(latent space)로 이동시키기 때문이다.
- 이러한 이유로 VAE는 학습이 끝난 후에 새로운 샘플을 가우시안 분포로부터 랜덤한 코딩을 샘플링해 디코딩해서 생성할 수 있다.

# Reference
- [08. 오토인코더 (AutoEncoder)](https://excelsior-cjh.tistory.com/187)
- [외부기고 새로운 인공지능 기술 GAN ② GAN의 개념과 이해](https://www.samsungsds.com/kr/insights/Generative-adversarial-network-AI-2.html)
- [Anomaly Detection 개요： [1] 이상치 탐지 분야에 대한 소개 및 주요 문제와 핵심 용어, 산업 현장 적용 사례 정리](https://hoya012.github.io/blog/anomaly-detection-overview-1/)
- [[정리노트] [AutoEncoder의 모든것] Chap2. Manifold Learning이란 무엇인가](https://deepinsight.tistory.com/124)