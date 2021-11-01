---
title: '[Deep Learning] CNN(ResNet50) tutorial'
description: ResNet50을 이용한 전이 학습 및 Custom model 튜토리얼
categories:
 - Deep Learning
tags: [Deep Learning, CNN, Convolution, Pooling, Transfer Learning, image classification]
mathjax: enable
---

# ResNet50
- 산의 이미지와 숲의 이미지 분류하는 문제
- [다운로드](https://ds-lecture-data.s3.ap-northeast-2.amazonaws.com/datasets/mountainForest.zip)
- 클래스당 약 350개의 이미지로 이루어져 있다.

## Import data
- [Keras ImageDataGenerator 참고](https://keras.io/api/preprocessing/image/)

```py
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.preprocessing import image

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model, Sequential

tr = '/content/drive/MyDrive/mountainForest/train'
val = '/content/drive/MyDrive/mountainForest/validation'

tr = tf.keras.preprocessing.image_dataset_from_directory(
    tr,
    labels="inferred",
    label_mode="binary",
    class_names=["forest", "mountain"],
    seed=6,
)

val = tf.keras.preprocessing.image_dataset_from_directory(
    val,
    labels="inferred",
    label_mode="binary",
    class_names=["forest", "mountain"],
    seed=6,
)

tr, val
'''
Found 533 files belonging to 2 classes.
Found 195 files belonging to 2 classes.
(<BatchDataset shapes: ((None, 256, 256, 3), (None, 1)), types: (tf.float32, tf.float32)>,
 <BatchDataset shapes: ((None, 256, 256, 3), (None, 1)), types: (tf.float32, tf.float32)>)
'''
```

## Instatiate Model

```py
# 기존 1000가지 클래스로의 분류문제를 풀 수 있는 ResNet 모델에서 Fully Connected layer 부분을 제거하는 역할을 한다.
resnet = ResNet50(weights='imagenet', include_top=False)

# ResNet50 레이어의 파라미터를 학습하지 않도록 설정한다.
# 이렇게 설정하면 역전파를 통해 오차 정보가 전파 되더라도 파라미터가 업데이트되지 않는다.
for layer in resnet.layers:
    layer.trainable = False

# 모델에 추가로 Fully connected layer를 추가한다.
# 이진 분류에 맞게 출력층을 설계한다.
x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(resnet.input, predictions)
```

## Fit Model

```py
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(tr, validation_data=val, epochs=5)
'''
Epoch 1/5
17/17 [==============================] - 155s 9s/step - loss: 0.1496 - accuracy: 0.9343 - val_loss: 0.0913 - val_accuracy: 0.9692
Epoch 2/5
17/17 [==============================] - 146s 9s/step - loss: 0.0039 - accuracy: 1.0000 - val_loss: 0.0353 - val_accuracy: 0.9949
Epoch 3/5
17/17 [==============================] - 147s 9s/step - loss: 0.0065 - accuracy: 0.9981 - val_loss: 0.1728 - val_accuracy: 0.9590
Epoch 4/5
17/17 [==============================] - 145s 9s/step - loss: 0.0122 - accuracy: 0.9944 - val_loss: 0.0560 - val_accuracy: 0.9795
Epoch 5/5
17/17 [==============================] - 147s 9s/step - loss: 0.0204 - accuracy: 0.9944 - val_loss: 0.0415 - val_accuracy: 0.9949
<keras.callbacks.History at 0x7fb07e1055d0>
'''


model.evaluate(val)
'''
7/7 [==============================] - 39s 5s/step - loss: 0.0415 - accuracy: 0.9949
[0.04146067425608635, 0.9948717951774597]
'''
```

# Custom CNN Model

## Make Model

```py
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(Flatten())
model2.add(Dense(64, activation='relu'))
model2.add(Dense(1, activation='softmax'))

model2.summary()
'''
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 254, 254, 32)      896       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 127, 127, 32)      0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 125, 125, 64)      18496     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 62, 62, 64)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 60, 60, 64)        36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 230400)            0         
_________________________________________________________________
dense_12 (Dense)             (None, 64)                14745664  
_________________________________________________________________
dense_13 (Dense)             (None, 1)                 65        
=================================================================
Total params: 14,802,049
Trainable params: 14,802,049
Non-trainable params: 0
_________________________________________________________________
'''
```

## Fit model

```py
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model2.fit(tr, validation_data=val, epochs=5)
'''
Epoch 1/5
17/17 [==============================] - 62s 4s/step - loss: 256.0300 - accuracy: 0.4841 - val_loss: 0.6757 - val_accuracy: 0.6564
Epoch 2/5
17/17 [==============================] - 61s 4s/step - loss: 0.2509 - accuracy: 0.4841 - val_loss: 1.3356 - val_accuracy: 0.6564
Epoch 3/5
17/17 [==============================] - 61s 4s/step - loss: 0.4466 - accuracy: 0.4841 - val_loss: 0.7562 - val_accuracy: 0.6564
Epoch 4/5
17/17 [==============================] - 61s 4s/step - loss: 0.1170 - accuracy: 0.4841 - val_loss: 0.2997 - val_accuracy: 0.6564
Epoch 5/5
17/17 [==============================] - 61s 4s/step - loss: 0.1087 - accuracy: 0.4841 - val_loss: 0.6576 - val_accuracy: 0.6564
<keras.callbacks.History at 0x7fb082f55c50>
'''


model2.evaluate(val)
'''
7/7 [==============================] - 6s 731ms/step - loss: 0.6576 - accuracy: 0.6564
[0.6576451063156128, 0.656410276889801]
'''
```

- ResNet의 성능이 참 좋다.