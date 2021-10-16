---
title: '[Deep Learning] TensorFlow & Keras Tutorial'
description: 텐서플로우와 케라스 기초 이론 및 실습
categories:
 - Deep Learning
tags: [Deep Learning, TensorFlow, Keras]
mathjax: enable
---

# TensorFlow / Keras
- ML 모델 개발하고 학습시키는 데 도움이 되는 핵심 오픈소스 라이브러리
- 2015년에 릴리즈 됐으며, 이는 딥러닝 세계의 관점에서 볼 때 꽤 오랜시간
- Keras는 사용자가 TF를 더 쉽고 편하게 사용할 수 있게 해주는 high level API 제공
- TF 2.x에서는 Keras를 딥러닝의 공식 API로 채택, Keras는 TF 내의 하나의 Framework로 개발

## import

```py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# MNIST dataset download
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
'''
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step
11501568/11490434 [==============================] - 0s 0us/step
'''

# Model 생성, compile
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Training / Evaluation
model.fit(x_train, y_train, epochs=10)

model.evaluate(x_test,  y_test)
'''
Epoch 1/10
1875/1875 [==============================] - 7s 2ms/step - loss: 0.2978 - accuracy: 0.9136
Epoch 2/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.1461 - accuracy: 0.9562
Epoch 3/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.1115 - accuracy: 0.9662
Epoch 4/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0888 - accuracy: 0.9725
Epoch 5/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0762 - accuracy: 0.9764
Epoch 6/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0672 - accuracy: 0.9790
Epoch 7/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0601 - accuracy: 0.9803
Epoch 8/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0551 - accuracy: 0.9815
Epoch 9/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0469 - accuracy: 0.9844
Epoch 10/10
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0445 - accuracy: 0.9853
313/313 [==============================] - 1s 3ms/step - loss: 0.0698 - accuracy: 0.9801
[0.0697677955031395, 0.9800999760627747]
'''
```

```py
idx = np.random.randint(len(x_train))
image = x_train[idx]


plt.imshow(image, cmap='gray')
plt.title(y_train[idx])
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/137588387-635e4f8e-55e5-4a15-9b48-42657c7b3830.png)


## Tensor
- multi-dimensional array를 나타내는 말로, TensorFlow의 기본 data type
- np array와 유사
- 자세한 사항 TIL 참조

# Dataset
- Data를 처리하여 model에 공급하기 위하여 TensorFlow에서는 `tf.data.Dataset`을 사용

## FashoinMNIST data

```py
mnist = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# train_images, train_labels의 shape 확인
print(train_images.shape, train_labels.shape)
'''
(60000, 28, 28) (60000,)
'''


# test_images, test_labels의 shape 확인
print(test_images.shape, test_labels.shape)
'''
(10000, 28, 28) (10000,)
'''


# training set의 각 class 별 image 수 확인
unique, counts = np.unique(train_labels, axis=-1, return_counts=True)
dict(zip(unique, counts))
'''
{0: 6000,
 1: 6000,
 2: 6000,
 3: 6000,
 4: 6000,
 5: 6000,
 6: 6000,
 7: 6000,
 8: 6000,
 9: 6000}
'''


# test set의 각 class 별 image 수 확인
unique, counts = np.unique(test_labels, axis=-1, return_counts=True)
dict(zip(unique, counts))
'''
{0: 1000,
 1: 1000,
 2: 1000,
 3: 1000,
 4: 1000,
 5: 1000,
 6: 1000,
 7: 1000,
 8: 1000,
 9: 1000}
'''
```

## Visualization

```py
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(class_names[train_labels[i]])
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/137588654-370487bc-2e03-4ff5-a7a4-5aa36c675cbc.png)

## 전처리
- image를 0~1사이 값으로 만들기 위하여 255로 나누어줌
- one-hot encoding: 다 0이고 정답만 1인 라벨을 만들어 주는 것
- 데이터를 다운받고 전처리를 하고 데이터셋을 만들어서 넣어 주는 부분을 만든것

```py
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.


train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)
```

## Make Dataset

```py
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
                buffer_size=100000).batch(64) # batch: 한번에 여러개
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(64) # testset은 셔플할 필요 없음


# Dataset을 통해 반복(iterate). 이미지와 정답(label)을 표시
imgs, lbs = next(iter(train_dataset)) # 하나만 뺄 수 있음
print(f"Feature batch shape: {imgs.shape}")
print(f"Labels batch shape: {lbs.shape}")

img = imgs[0]
lb = lbs[0]
plt.imshow(img, cmap='gray')
plt.show()
print(f"Label: {lb}")
'''
Feature batch shape: (64, 28, 28)
Labels batch shape: (64, 10)
Label: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
'''
```

![image](https://user-images.githubusercontent.com/79494088/137588866-26e1e08c-3ae9-416f-9cca-aea57a801552.png)

### Custom Dataset

```py
a = np.arange(10)
print(a)

ds_tensors = tf.data.Dataset.from_tensor_slices(a)
print(ds_tensors)

for x in ds_tensors:
    print (x)
'''
[0 1 2 3 4 5 6 7 8 9]
<TensorSliceDataset shapes: (), types: tf.int64>
tf.Tensor(0, shape=(), dtype=int64)
tf.Tensor(1, shape=(), dtype=int64)
tf.Tensor(2, shape=(), dtype=int64)
tf.Tensor(3, shape=(), dtype=int64)
tf.Tensor(4, shape=(), dtype=int64)
tf.Tensor(5, shape=(), dtype=int64)
tf.Tensor(6, shape=(), dtype=int64)
tf.Tensor(7, shape=(), dtype=int64)
tf.Tensor(8, shape=(), dtype=int64)
tf.Tensor(9, shape=(), dtype=int64)
'''


# data 전처리(변환), shuffle, batch 추가
ds_tensors = ds_tensors.map(tf.square).shuffle(10).batch(2)


for _ in range(3): # 에폭
    for x in ds_tensors:
        print(x)
    print('='*50)
'''
tf.Tensor([ 9 36], shape=(2,), dtype=int64)
tf.Tensor([49 81], shape=(2,), dtype=int64)
tf.Tensor([64  4], shape=(2,), dtype=int64)
tf.Tensor([1 0], shape=(2,), dtype=int64)
tf.Tensor([16 25], shape=(2,), dtype=int64)
==================================================
tf.Tensor([0 9], shape=(2,), dtype=int64)
tf.Tensor([49  4], shape=(2,), dtype=int64)
tf.Tensor([16  1], shape=(2,), dtype=int64)
tf.Tensor([64 81], shape=(2,), dtype=int64)
tf.Tensor([36 25], shape=(2,), dtype=int64)
==================================================
tf.Tensor([36  0], shape=(2,), dtype=int64)
tf.Tensor([4 1], shape=(2,), dtype=int64)
tf.Tensor([49 64], shape=(2,), dtype=int64)
tf.Tensor([ 9 81], shape=(2,), dtype=int64)
tf.Tensor([16 25], shape=(2,), dtype=int64)
==================================================
'''
```

# Model

## Keras Sequential API
- 가장 쉽고 기본적인 방법이며 사용할 수만 있다면 굳이 다른 것을 사용할 필요 없이 이것을 사용하는게 좋다.

```py
def create_seq_model():
    model = keras.Sequential() # 선언 후 벽돌 쌓듯 레이어를 추가
    model.add(keras.layers.Flatten(input_shape=(28, 28))) # 벡터로 펴져야 들어갈 수 있음. 첫번째는 반드시 input shape을 써야함
    model.add(keras.layers.Dense(128, activation='relu')) # 128개로 구성된 MLP 구현
    model.add(keras.layers.Dropout(0.2)) # 오버피팅 막아주는 테크닉
    model.add(keras.layers.Dense(10, activation='softmax')) # 실제 레이어는 두 개
    return model


seq_model = create_seq_model()


seq_model.summary() # None: batch size가 들어갈 자리
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_1 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               100480    
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
'''
```

## Keras Functional API
- Sequential API 보다 다양한 Network를 만들 수 있다.

```py
def create_func_model():
    inputs = keras.Input(shape=(28,28))
    flatten = keras.layers.Flatten()(inputs)
    dense = keras.layers.Dense(128, activation='relu')(flatten)
    drop = keras.layers.Dropout(0.2)(dense)
    outputs = keras.layers.Dense(10, activation='softmax')(drop)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

func_model = create_func_model()

func_model.summary()
'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 28, 28)]          0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 128)               100480    
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 10)                1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
'''
```

## Model Class Subclassing

```py
class SubClassModel(keras.Model):
    def __init__(self):
        super(SubClassModel, self).__init__()
        self.flatten = keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.drop = keras.layers.Dropout(0.2)
        self.dense2 = keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop(x)
        return self.dense2(x)


subclass_model = SubClassModel()


inputs = tf.zeros((1, 28, 28))
subclass_model(inputs)
subclass_model.summary()
'''
Model: "sub_class_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_3 (Flatten)          multiple                  0         
_________________________________________________________________
dense_6 (Dense)              multiple                  100480    
_________________________________________________________________
dropout_3 (Dropout)          multiple                  0         
_________________________________________________________________
dense_7 (Dense)              multiple                  1290      
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
'''
```

- 가상의 data 만들어서 예측

```py
inputs = tf.random.normal((1, 28, 28))
outputs = subclass_model(inputs)
pred = tf.argmax(outputs, -1)
print(f"Predicted class: {pred}")
'''
Predicted class: [7]
'''
```

# Traning / Validation

## Keras API

```py
learning_rate = 0.001 
seq_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = seq_model.fit(train_dataset, epochs=10, validation_data=test_dataset)
'''
Epoch 1/10
938/938 [==============================] - 4s 4ms/step - loss: 0.5521 - accuracy: 0.8080 - val_loss: 0.4460 - val_accuracy: 0.8390
Epoch 2/10
938/938 [==============================] - 4s 4ms/step - loss: 0.4031 - accuracy: 0.8572 - val_loss: 0.3983 - val_accuracy: 0.8530
Epoch 3/10
938/938 [==============================] - 4s 4ms/step - loss: 0.3700 - accuracy: 0.8657 - val_loss: 0.3762 - val_accuracy: 0.8657
Epoch 4/10
938/938 [==============================] - 4s 4ms/step - loss: 0.3486 - accuracy: 0.8734 - val_loss: 0.3723 - val_accuracy: 0.8623
Epoch 5/10
938/938 [==============================] - 4s 4ms/step - loss: 0.3343 - accuracy: 0.8783 - val_loss: 0.3474 - val_accuracy: 0.8768
Epoch 6/10
938/938 [==============================] - 4s 4ms/step - loss: 0.3206 - accuracy: 0.8827 - val_loss: 0.3452 - val_accuracy: 0.8763
Epoch 7/10
938/938 [==============================] - 4s 4ms/step - loss: 0.3092 - accuracy: 0.8859 - val_loss: 0.3463 - val_accuracy: 0.8758
Epoch 8/10
938/938 [==============================] - 4s 4ms/step - loss: 0.3005 - accuracy: 0.8890 - val_loss: 0.3339 - val_accuracy: 0.8808
Epoch 9/10
938/938 [==============================] - 4s 4ms/step - loss: 0.2912 - accuracy: 0.8922 - val_loss: 0.3293 - val_accuracy: 0.8821
Epoch 10/10
938/938 [==============================] - 4s 4ms/step - loss: 0.2817 - accuracy: 0.8954 - val_loss: 0.3298 - val_accuracy: 0.8811
'''


## Plot losses
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/137589211-d6bf1277-2969-4b46-ba66-591921746736.png)


```py
## Plot Accuracy
plt.plot(history.history['accuracy'], 'b-', label='acc')
plt.plot(history.history['val_accuracy'], 'r--', label='val_acc')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/137589221-786a494f-d952-469d-9c79-b460ba494b8e.png)


## GradientTape

```py
# loss function
loss_object = keras.losses.CategoricalCrossentropy()


# optimizer
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)


# loss, accuracy 계산
train_loss = keras.metrics.Mean(name='train_loss')
train_accuracy = keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = keras.metrics.Mean(name='test_loss')
test_accuracy = keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function # 그래프 모드: 속도가 빨라짐. 디버깅이 필요할 때 떼고 하면 됨
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(model, images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 10

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_dataset:
        train_step(func_model, images, labels)

    for test_images, test_labels in test_dataset:
        test_step(func_model, test_images, test_labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
'''
Epoch 1, Loss: 0.5537072420120239, Accuracy: 80.80166625976562, Test Loss: 0.4622400999069214, Test Accuracy: 82.79000091552734
Epoch 2, Loss: 0.4053775370121002, Accuracy: 85.4433364868164, Test Loss: 0.38666844367980957, Test Accuracy: 86.47000122070312
Epoch 3, Loss: 0.3678165376186371, Accuracy: 86.74500274658203, Test Loss: 0.38237690925598145, Test Accuracy: 86.11000061035156
Epoch 4, Loss: 0.3475690484046936, Accuracy: 87.2933349609375, Test Loss: 0.3730221092700958, Test Accuracy: 87.0199966430664
Epoch 5, Loss: 0.33085232973098755, Accuracy: 87.90166473388672, Test Loss: 0.3561401963233948, Test Accuracy: 87.06999969482422
Epoch 6, Loss: 0.3185732960700989, Accuracy: 88.27000427246094, Test Loss: 0.34758004546165466, Test Accuracy: 87.51000213623047
Epoch 7, Loss: 0.3109299838542938, Accuracy: 88.59000396728516, Test Loss: 0.3540751338005066, Test Accuracy: 87.30999755859375
Epoch 8, Loss: 0.29884836077690125, Accuracy: 88.96499633789062, Test Loss: 0.33863919973373413, Test Accuracy: 87.69999694824219
Epoch 9, Loss: 0.2917931377887726, Accuracy: 89.1683349609375, Test Loss: 0.32866981625556946, Test Accuracy: 88.06999969482422
Epoch 10, Loss: 0.28563857078552246, Accuracy: 89.35333251953125, Test Loss: 0.3342178463935852, Test Accuracy: 87.84000396728516
'''
```

# Model Import / Export

## parameter만

```py
seq_model.save_weights('seq_model.ckpt')


seq_model_2 = create_seq_model()
seq_model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


seq_model_2.evaluate(test_dataset)
'''
157/157 [==============================] - 1s 3ms/step - loss: 2.4642 - accuracy: 0.1115
[2.464158773422241, 0.11150000244379044]
'''


seq_model_2.load_weights('seq_model.ckpt')
'''
<tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7fae85deb890>
'''


seq_model_2.evaluate(test_dataset)
'''
157/157 [==============================] - 1s 3ms/step - loss: 0.3298 - accuracy: 0.8811
[0.3297693133354187, 0.8810999989509583]
'''
```

## Model 전체

```py
seq_model.save('seq_model')
'''
INFO:tensorflow:Assets written to: seq_model/assets
'''


!ls
'''
checkpoint   seq_model				 seq_model.ckpt.index
sample_data  seq_model.ckpt.data-00000-of-00001
'''


seq_model_3 = keras.models.load_model('seq_model')


seq_model_3.evaluate(test_dataset)
'''
157/157 [==============================] - 1s 3ms/step - loss: 0.3298 - accuracy: 0.8811
[0.3297693133354187, 0.8810999989509583]
'''
```

# Tensorboard

## Keras Callback

```py
%load_ext tensorboard


new_model_1 = create_seq_model()
new_model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


new_model_1.evaluate(test_dataset)
'''
157/157 [==============================] - 1s 3ms/step - loss: 2.2566 - accuracy: 0.1386
[2.2565832138061523, 0.13860000669956207]
'''


log_dir = './logs/new_model_1'

tensorboard_cb = keras.callbacks.TensorBoard(log_dir, histogram_freq=1)


new_model_1.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset,
          callbacks=[tensorboard_cb])
'''
Epoch 1/10
938/938 [==============================] - 4s 4ms/step - loss: 0.5577 - accuracy: 0.8051 - val_loss: 0.4447 - val_accuracy: 0.8408
Epoch 2/10
938/938 [==============================] - 4s 4ms/step - loss: 0.4093 - accuracy: 0.8537 - val_loss: 0.4102 - val_accuracy: 0.8533
Epoch 3/10
938/938 [==============================] - 4s 4ms/step - loss: 0.3732 - accuracy: 0.8651 - val_loss: 0.3724 - val_accuracy: 0.8680
Epoch 4/10
938/938 [==============================] - 4s 4ms/step - loss: 0.3488 - accuracy: 0.8722 - val_loss: 0.3730 - val_accuracy: 0.8659
Epoch 5/10
938/938 [==============================] - 4s 4ms/step - loss: 0.3329 - accuracy: 0.8789 - val_loss: 0.3684 - val_accuracy: 0.8670
Epoch 6/10
938/938 [==============================] - 4s 4ms/step - loss: 0.3204 - accuracy: 0.8846 - val_loss: 0.3486 - val_accuracy: 0.8755
Epoch 7/10
938/938 [==============================] - 4s 4ms/step - loss: 0.3095 - accuracy: 0.8852 - val_loss: 0.3445 - val_accuracy: 0.8785
Epoch 8/10
938/938 [==============================] - 4s 4ms/step - loss: 0.2996 - accuracy: 0.8884 - val_loss: 0.3589 - val_accuracy: 0.8683
Epoch 9/10
938/938 [==============================] - 4s 4ms/step - loss: 0.2923 - accuracy: 0.8920 - val_loss: 0.3410 - val_accuracy: 0.8766
Epoch 10/10
938/938 [==============================] - 4s 4ms/step - loss: 0.2844 - accuracy: 0.8948 - val_loss: 0.3455 - val_accuracy: 0.8798
<keras.callbacks.History at 0x7fae8428ce50>
'''


%tensorboard --logdir $log_dir
```

![image](https://user-images.githubusercontent.com/79494088/137589421-5e3fad99-3456-4ffb-8f15-37413c2eee94.png)

## Summary Writer

```py
new_model_2 = create_seq_model()


# loss function
loss_object = keras.losses.CategoricalCrossentropy()


# optimizer
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)


# loss, accuracy 계산
train_loss = keras.metrics.Mean(name='train_loss')
train_accuracy = keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = keras.metrics.Mean(name='test_loss')
test_accuracy = keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(model, images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


EPOCHS = 10

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_dataset:
        train_step(new_model_2, images, labels)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for test_images, test_labels in test_dataset:
        test_step(new_model_2, test_images, test_labels)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
'''
Epoch 1, Loss: 0.5432561039924622, Accuracy: 80.99832916259766, Test Loss: 0.43092185258865356, Test Accuracy: 84.72999572753906
Epoch 2, Loss: 0.40512388944625854, Accuracy: 85.47000122070312, Test Loss: 0.39184921979904175, Test Accuracy: 85.8499984741211
Epoch 3, Loss: 0.3681531548500061, Accuracy: 86.66832733154297, Test Loss: 0.36701878905296326, Test Accuracy: 86.94000244140625
Epoch 4, Loss: 0.3476184904575348, Accuracy: 87.28666687011719, Test Loss: 0.3711545467376709, Test Accuracy: 86.56999969482422
Epoch 5, Loss: 0.32993197441101074, Accuracy: 87.89666748046875, Test Loss: 0.34852442145347595, Test Accuracy: 87.08999633789062
Epoch 6, Loss: 0.3175106942653656, Accuracy: 88.40833282470703, Test Loss: 0.3412656784057617, Test Accuracy: 87.87000274658203
Epoch 7, Loss: 0.3032893240451813, Accuracy: 88.74666595458984, Test Loss: 0.3460729718208313, Test Accuracy: 87.31999969482422
Epoch 8, Loss: 0.2971169650554657, Accuracy: 88.8800048828125, Test Loss: 0.3324110507965088, Test Accuracy: 87.91999816894531
Epoch 9, Loss: 0.2880474627017975, Accuracy: 89.29332733154297, Test Loss: 0.33137768507003784, Test Accuracy: 88.02000427246094
Epoch 10, Loss: 0.2817001938819885, Accuracy: 89.53833770751953, Test Loss: 0.33480119705200195, Test Accuracy: 87.87999725341797
'''


%tensorboard --logdir 'logs/gradient_tape'
```

![image](https://user-images.githubusercontent.com/79494088/137589492-81fddac7-4f42-4843-9f61-17b1455a2819.png)
