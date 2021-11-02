---
title: '[Deep Learning] Image Segmentation & Data Augmentation'
description: Image Segmentation 개념과 대표 모델, Image Augmentation의 개념과 기본적인 증강방식 활용, Object Recognition 개념과 활용
categories:
 - Deep Learning
tags: [Deep Learning, Image Segmentation, Image Augmentation, 증강, Object Recognition]
mathjax: enable
---

# Warm Up
- [Lecture 11 Detection and Segmentation](https://youtu.be/nDPWywWRIRo)
- [Unet 튜토리얼](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

# Image Segmentation(이미지 분할)
- 이미지 분할을 위해 이미지를 픽셀 단위의 마스크를 정보를 출력하도록 신경망을 훈련시킬 것이다.
- 이미지 분할은 의료 영상, 자율 주행차, 보안, 위성, 항공사진 등의 분야에서 응용되고 있다.
- 가장 전통적인 방식의 이미지 분할을 FCN이라고 하여 고양이 그림이 있는 형태로 구성되며, 비교적 최근에 개발된 Unet 이후에는 대부분 Unet이 이미지 분할에서 사용된다.

<img src="https://chadrick-kwag.net/wp-content/uploads/2020/09/fwfewfewfewq.png">

<img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png"> 

- 튜토리얼 사용 데이터셋: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
    * class 1 : 애완동물이 속한 픽셀
    * class 2 : 애완동물과 인접한 픽셀
    * class 3 : 위에 속하지 않는 경우/주변 픽셀

```py
!pip install git+https://github.com/tensorflow/examples.git
!pip install -U tfds-nightly


import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt
```

## Download Dataset

```py
# download
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


# 이미지를 뒤집는 간단한 증강
# 영상이 0, 1로 정규화
# 분할 마스크 픽셀에 1, 2, 3이라는 레이블 할당
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask


@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
# data augmentation
  if tf.random.uniform(()) > 0.5: # 50% 확률로... (0~1 사이값 중에서 절반)
    input_image = tf.image.flip_left_right(input_image) # 좌우반전
    input_mask = tf.image.flip_left_right(input_mask)   # 레이블도 좌우반전

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


# 동일한 분할 지속 사용
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE # // 나누기 후 소수점 버림

STEPS_PER_EPOCH
'''
57
'''


train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE) # auto
test = dataset['test'].map(load_image_test)


train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)


# 이미지 예제와 데이터셋 대응 마스크
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


for image, mask in train.take(100):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])
```

![image](https://user-images.githubusercontent.com/79494088/139678734-5576d333-5351-411b-bdce-fc99eb8de523.png)

## 모델 정의
- 사용하는 모델은 수정된 Unet이다.
- Unet은 인코더(Down Sampler)와 디코더(Up Sampler)를 포함한다.
- 강력한 기능을 학습하고 훈련 가능한 매개변수의 수를 줄이기 위해 미리 훈련된 모델을 인코더로 사용할 수 있다.
- 이번 튜토리얼의 인코더는 미리 훈룐된 MobileNetV2 모델이 될 것이며 이 모델의 중간 출력이 사용될 것이다.
- 디코더는 [Pix2pix 튜토리얼](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py)의 TF 예제에서 이미 구현한 업샘플 블록이 될 것이다.
- 3개의 채널을 출력하는 이유는 픽셀당 3개의 가능한 라벨이 있기 때문이다.
- 이것을 각 화소가 세 개의 class로 분류되는 다중 분류로 생각하면 된다.

```py
OUTPUT_CHANNELS = 3
```

- 인코더는 모델의 중간층에서 나오는 특정 출력으로 구성된다.
- 인코더는 교육 과정 중에 학습되지 않는다.

```py
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False) # 기존 인풋사이즈 재사용 False

#이 층들의 활성화를 이용합시다
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# 특징추출 모델을 만듭시다
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False
'''
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
9412608/9406464 [==============================] - 0s 0us/step
'''


# 디코더 / 업샘플러
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

# Upsample : https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py


def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  x = inputs

  # 모델을 통해 다운샘플링합시다
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # 건너뛰기 연결을 업샘플링하고 설정하세요
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # 이 모델의 마지막 층입니다
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
```

## 모델 훈련
- 여기서 사용되고 있는 손실 함수는 loss.sparse_categorical_crossentropy이다.
- 이 손실 함수를 사용하는 이유는 네트워크가 멀티 클래스 예측과 마찬가지로 픽셀마다 레이블을 할당하려고 하기 때문이다.
- 실제 분할 마스크에서 각 픽셀은 0, 1, 2를 갖고 있다.
- 이 곳의 네트워크는 세 개의 채널을 출력하고 있다.
- 기본적으로 각 채널은 클래스를 예측하는 방법을 배우려고 하고 있으며, 위 손실 함수는 그러한 시나리오에 권장되는 손실이다.
- 네트워크의 출력을 사용하여 픽셀에 할당된 레이블은 가장 높은 값을 가진 채널이다.
- 이 것이 create_mask 함수가 하는 일이다.

```py
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# 모델 구조 확인
tf.keras.utils.plot_model(model, show_shapes=True)
```

![image](https://user-images.githubusercontent.com/79494088/139683464-3a1ac8fc-3e6a-4fa5-8790-d4feb6ac600f.png)

```py
# 모델을 시험해보고 훈련전의 예측 확인
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]


def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])


show_predictions()
```

![image](https://user-images.githubusercontent.com/79494088/139683746-4a739f2f-f412-4c21-b235-18f84c6cab43.png)

```py
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\n에포크 이후 예측 예시 {}\n'.format(epoch+1))


EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])
'''
에포크 이후 예측 예시 20
'''
```

![image](https://user-images.githubusercontent.com/79494088/139683857-e67d13ab-d6c9-4827-bc31-d5d54ca87669.png)

```py
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/139683984-2eb95b36-bc04-4f16-82a2-a237c6516013.png)

## 예측
- 시간을 절약하기 위해 에포크 수를 작게 유지했지만, 보다 정확한 결과를 얻기 위해 에포크를 더 높게 설정할 수 있다.

```py
show_predictions(test_dataset, 3)
```

## Reference
- [텐서플로 튜토리얼 공식 영문 문서](https://www.tensorflow.org/?hl=en)
- [tensorflow/docs-l10n](https://github.com/tensorflow/docs-l10n) 

# Data Augmentation(데이터 증강)
- 이 튜토리얼에서는 이미지 회전과 같은 무작위 변환을 적용하여 훈련 세트의 다양성을 증가시키는 기술인 데이터 증강의 예를 보여준다.
- 두가지 방법으로 데이터 증강을 적용한다.
- 먼저, [Keras 전처리 레이어](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/)를 사용하고, 그 다음으로 tf.image를 사용한다.

```py
!pip install tf-nightly


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
```

## Download Dataset

- 이 튜토리얼에서는 [tf_flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) 데이터셋을 사용한다.
- 편의를 위해 [TensorFlow Datasets](https://www.tensorflow.org/datasets)를 사용하여 데이터셋을 다운로드한다.
- 데이터를 가져오는 다른 방법: [이미지 로드](https://www.tensorflow.org/tutorials/load_data/images)

```py
# Download dataset
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)


# 꽃 데이터셋에는 5개의 class가 있다.
num_classes = metadata.features['label'].num_classes
print(num_classes)
'''
5
'''


# 데이터셋에서 이미지를 검색하고 이를 사용하여 데이터 증강을 수행한다.
get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
```

![image](https://user-images.githubusercontent.com/79494088/139690245-5abc2a98-4a35-4ad5-a44b-fee170a57259.png)

## Keras 전처리 레이어 사용
- [Keras 전처리 레이어](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing)는 현재 실험적 단계이다.

### 크기 및 배율 조정
- 전처리 레이어를 사용하여 이미지를 일관된 모양으로 [크기 조정](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Resizing)하고 픽셀 값의 [배율을 조정](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Rescaling)할 수 있다.

```py
IMG_SIZE = 180

resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255)
])
```

- 위의 배율 조정 레이어는 픽셀 값을 `[0,1]`로 표준화한다.
- 그렇지 않고 `[-1,1]`을 원할 경우, `Rescaling(1./127.5, offset=-1)`을 작성하면 된다.

```py
# 이미지 적용 결과
result = resize_and_rescale(image)
_ = plt.imshow(result)
```

![image](https://user-images.githubusercontent.com/79494088/139690653-252b4134-f8ac-4b66-a123-315e5eda14e5.png)

```py
# 픽셀 0-1에 있는지 확인
print("Min and max pixel values:", result.numpy().min(), result.numpy().max())
'''
Min and max pixel values: 0.0 1.0
'''
```

### 데이터 증강
- 데이터 증강에도 전처리 레이어를 사용할 수 있다.
- 몇 개의 전처리 레이어를 만들어 동일한 이미지에 반복적으로 적용한다.

```py
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

# https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing


# Add the image to a batch
image = tf.expand_dims(image, 0)


plt.figure(figsize=(10, 10))
for i in range(9):
  augmented_image = data_augmentation(image)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0])
  plt.axis("off")
```

![image](https://user-images.githubusercontent.com/79494088/139690955-89a1446b-cfd5-47b0-88cf-225d60396a9d.png)

- `layers.RandomContrast`, `layers.RandomCrop`, `layers.RandomZoom` 등 데이터 증강에 사용할 수 있는 다양한 전처리 [레이어](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing)가 있다.

### 전처리 레이어 두 가지 옵션

#### 1. 전처리 레이어를 모델의 일부로 만들기

```py
model = tf.keras.Sequential([
  resize_and_rescale,
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # Rest of your model
])
```

- 유의해야 할 두가지 중요한 사항
    - 데이터 증강은 나머지 레이어와 동기적으로 기기에서 실행되며 GPU 가속을 이용한다.
    - `model.save`를 사용하여 모델을 내보낼 때 전처리 레이어가 모델의 나머지 부분과 함께 저장된다. 나중에 이 모델을 배포하면 레이어 구성에 따라 이미지가 자동으로 표준화된다. 이를 통해 서버측 논리를 다시 구현해야 하는 노력을 덜 수 있다.
- 데이터 증강은 테스트할 때 비활성화 되므로 입력 이미지는 `model.fit` 호출 중에만 증강된다.

#### 2. 데이터 세트에 전처리 레이어 적용

```py
aug_ds = train_ds.map(
  lambda x, y: (resize_and_rescale(x, training=True), y))
```

- 이 접근 방식에서는 `Dataset.map`을 사용하여 증강 이미지 배치를 생성하는 데이터셋을 만든다.
    - 데이터 증강은 CPU에서 비동기적으로 이루어지며 차단되지 않는다.
    - `Dataset.prefech`를 사용하여 GPU에서 모델 훈련을 데이터 전처리와 중첩할 수 있다.
    - 이 경우, 전처리 레이어는 `model.save`를 호출할 때 모델과 함께 내보내지지 않는다.
    - 저장하기 전에 이 레이어를 모델에 연결하거나 서버측에서 다시 구현해야 한다.

### 전처리 레이어 적용
- 위에서 생성한 전처리 레이어로 훈련, 검증 및 테스트 데이터셋을 구성한다.
- 병렬 읽기 및 버퍼링된 프리패치를 사용하여 I/O 차단 없이 디스크에서 배치를 생성하여 성능을 높이도록 데이터셋을 구성한다.
- 데이터 증강은 훈련 세트에만 적용해야 한다.

```py
batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

def prepare(ds, shuffle=False, augment=False):
  # Resize and rescale all datasets
  ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
              num_parallel_calls=AUTOTUNE)

  if shuffle:
    ds = ds.shuffle(1000)

  # Batch all datasets
  ds = ds.batch(batch_size)

  # Use data augmentation only on the training set
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefecting on all datasets
  return ds.prefetch(buffer_size=AUTOTUNE)


train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)
```

### 모델 훈련

```py
model = tf.keras.Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


epochs=5
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
'''
Epoch 1/5
92/92 [==============================] - 45s 128ms/step - loss: 1.4877 - accuracy: 0.3378 - val_loss: 1.2423 - val_accuracy: 0.5204
Epoch 2/5
92/92 [==============================] - 10s 101ms/step - loss: 1.1254 - accuracy: 0.5490 - val_loss: 1.0362 - val_accuracy: 0.5995
Epoch 3/5
92/92 [==============================] - 10s 102ms/step - loss: 1.0058 - accuracy: 0.5877 - val_loss: 0.9426 - val_accuracy: 0.6757
Epoch 4/5
92/92 [==============================] - 10s 102ms/step - loss: 0.9304 - accuracy: 0.6285 - val_loss: 0.9565 - val_accuracy: 0.6322
Epoch 5/5
92/92 [==============================] - 10s 102ms/step - loss: 0.8918 - accuracy: 0.6547 - val_loss: 0.8652 - val_accuracy: 0.6785
'''


loss, acc = model.evaluate(test_ds)
print("Accuracy", acc)
'''
12/12 [==============================] - 1s 42ms/step - loss: 0.9024 - accuracy: 0.6431
Accuracy 0.6430517435073853
'''
```


### 사용자 정의 데이터 증강

```py
def random_invert_img(x, p=0.5):
  if  tf.random.uniform([]) < p:
    x = (255-x)
  else:
    x
  return x


def random_invert(factor=0.5):
  return layers.Lambda(lambda x: random_invert_img(x, factor))

random_invert = random_invert()


plt.figure(figsize=(10, 10))
for i in range(9):
  augmented_image = random_invert(image)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0].numpy().astype("uint8"))
  plt.axis("off")
```

![image](https://user-images.githubusercontent.com/79494088/139692454-763c37e8-466e-4cb8-b754-5ba07ec67fdb.png)

```py
# 서브 클래스 생성을 통해 사용자 정의 레이어 구현
class RandomInvert(layers.Layer):
  def __init__(self, factor=0.5, **kwargs):
    super().__init__(**kwargs)
    self.factor = factor

  def call(self, x):
    return random_invert_img(x)


_ = plt.imshow(RandomInvert()(image)[0])
```

![image](https://user-images.githubusercontent.com/79494088/139692614-05e8f309-09f2-44b4-b588-b79d3ae96dad.png)

## tf.image 사용
- 위 `layers.preprocessing` 유틸리티는 편리하다.
- 보다 세밀한 제어를 위해 `tf.data` 및 `tf.image`를 사용하여 고유한 데이터 증강 파이프라인 또는 레이어를 작성할 수 있다.
- [TensorFlow 애드온 이미지: 작업](https://www.tensorflow.org/addons/tutorials/image_ops), [TensorFlow I/O: 색 공간 변환](https://www.tensorflow.org/io/tutorials/colorspace)

```py
# 새로 시작
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)


# 적절한 이미지 검색
image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
```

![image](https://user-images.githubusercontent.com/79494088/139775667-b90a930d-ffd8-451f-83dc-73fc3250f60e.png)

```py
# 원본 이미지와 증강이미지 비교
def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)
```

### 데이터 증강

#### 이미지 뒤집기

```py
flipped = tf.image.flip_left_right(image)
visualize(image, flipped)
```

![image](https://user-images.githubusercontent.com/79494088/139775833-512881be-9388-44ef-8788-d99ce7520e41.png)

#### 이미지 회색조

```py
grayscaled = tf.image.rgb_to_grayscale(image)
visualize(image, tf.squeeze(grayscaled))
_ = plt.colorbar()
```

![image](https://user-images.githubusercontent.com/79494088/139775901-901eab06-04d0-4018-ad4a-d2a927ed77d8.png)

#### 이미지 포화

```py
# 채도 계수 제공
saturated = tf.image.adjust_saturation(image, 3)
visualize(image, saturated)
```

![image](https://user-images.githubusercontent.com/79494088/139775954-5fa294db-2e04-458d-8e33-027a1093b5ee.png)

#### 이미지 밝게 변경

```py
bright = tf.image.adjust_brightness(image, 0.4)
visualize(image, bright)
```

![image](https://user-images.githubusercontent.com/79494088/139776035-bdc3175c-9680-4937-af3c-297228afb3e5.png)

#### 이미지 중앙 자르기

```py
cropped = tf.image.central_crop(image, central_fraction=0.5)
visualize(image,cropped)
```

![image](https://user-images.githubusercontent.com/79494088/139776081-cd6fa4b2-2a34-43b9-ac55-d317db7458b5.png)

#### 이미지 회전

```py
rotated = tf.image.rot90(image)
visualize(image, rotated)
```

![image](https://user-images.githubusercontent.com/79494088/139776118-fac5b84d-c83b-456e-8a16-799b7868777a.png)

### 데이터셋에 증강 적용

```py
def resize_and_rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image = (image / 255.0)
  return image, label


def augment(image,label):
  image, label = resize_and_rescale(image, label)
  # Add 6 pixels of padding
  image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6) 
   # Random crop back to the original size
  image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
  image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
  image = tf.clip_by_value(image, 0, 1)
  return image, label


# 데이터셋 구성
train_ds = (
    train_ds
    .shuffle(1000)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
) 


val_ds = (
    val_ds
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)


test_ds = (
    test_ds
    .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)
```

# Object Recognition(객채 인식)
- 사전 학습된 이미지 분류 신경망을 사용하여 다중 객체를 검출한다.
- CNN의 가운데 레이어 출력 결과인 피쳐 맵 위에 슬라이딩 윈도우를 올린다.
- 사전 학습된 신경망은 해당 이미지의 클래스를 반환한다.
- 만약 이미지에서 서로 다른 객체가 검출되면 확률값을 둘로 쪼개서 반환한다.
- 하나의 객체만 나오더라도 충분한 확률값이 아니면 분할하여 반환한다.
- 다중객체를 찾는 방법의 하나는 이미지 위에 슬라이딩 윈도우를 올리고 윈도우 내에서 단일 객체 검출을 시도하는 것이다.
- 이미지를 224 * 224로 다운 샘플링하는 대신에 원본을 두배로 올려 448 * 448로 샘플링한다.
- 그 다음 crop된 이미지를 만들어 분류기에 넣는다.

```py
from keras.applications import vgg16
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Input, TimeDistributed
import numpy as np
from collections import Counter, defaultdict
from keras.preprocessing import image
from PIL import ImageDraw

from scipy.misc import imread, imresize, imsave, fromimage, toimage

try:
    from io import BytesIO
except ImportError:
    from StringIO import StringIO as BytesIO
import PIL
from IPython.display import clear_output, Image, display, HTML


def showarray(a, fmt='jpeg'):
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def preprocess_image(image_path, target_size=None):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

def deprocess_image(x, w, h):
    x = x.copy()
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, w, h))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((w, h, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


base_model = vgg16.VGG16(weights='imagenet', include_top=True)
base_model.summary()
'''
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
553467904/553467096 [==============================] - 9s 0us/step
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000   
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
'''


cat_dog = preprocess_image('cat_dog.jpg', target_size=(224, 224))
preds = base_model.predict(cat_dog)
print('Predicted:', vgg16.decode_predictions(preds, top=3)[0])
'''
Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
40960/35363 [==================================] - 0s 0us/step
Predicted: [('n02099601', 'golden_retriever', 0.09932627), ('n02100735', 'English_setter', 0.053414267), ('n02106662', 'German_shepherd', 0.04876063)]
'''


cat_dog_img = image.load_img('cat_dog.jpg', target_size=(448, 448))
draw = ImageDraw.Draw(cat_dog_img)
draw.rectangle((192, 96, 416, 320), outline=(0, 0, 0))
draw.rectangle((0, 192, 224, 416), outline=(0, 0, 0))
cat_dog_img
```

![image](https://user-images.githubusercontent.com/79494088/139777378-4bd6ae48-e300-4aac-9d36-bc3cea35cd5e.png)

```py
# 이미지 확대
cat_dog2 = preprocess_image('cat_dog.jpg', target_size=(448, 448))
showarray(deprocess_image(cat_dog2, 448, 448))
```

![image](https://user-images.githubusercontent.com/79494088/139777526-7d28bd0f-aa1f-4a09-8d05-497a8151c763.png)

```py
# 여러 구획으로 나눠준다.
crops = []
for x in range(7):
    for y in range(7):
        crops.append(cat_dog2[0, x * 32: x * 32 + 224, y * 32: y * 32 + 224, :])
crops = np.asarray(crops)
showarray(deprocess_image(crops[0], 224, 224))
```

![image](https://user-images.githubusercontent.com/79494088/139777589-e4edb007-00e2-4531-a3e9-61b116d58857.png)

```py
# 각각의 이미지를 분류기에 넣고 어떤 클래스가 나오는 지 확인
preds = base_model.predict(vgg16.preprocess_input(crops))
crop_scores = defaultdict(list)
for idx, pred in enumerate(vgg16.decode_predictions(preds, top=1)):
    _, label, weight = pred[0]
    crop_scores[label].append((idx, weight))
crop_scores.keys()
'''
dict_keys(['Labrador_retriever', 'Norwegian_elkhound', 'tiger_cat', 'Egyptian_cat', 'tabby', 'kuvasz', 'flat-coated_retriever', 'standard_schnauzer'])
'''
```

- 여러 이미지에서 개나 고양이가 검출됐다.
- 그러나 여러개의 품종으로 검출된 것을 볼 수 있다.
- 여러개의 레이블 중 가장 높은 확률의 이미지를 찾는 것을 실행해보자.

```py
crops.shape
'''
(49, 224, 224, 3)
'''


def best_image_for_label(l, label):
    idx = max(l[label], key=lambda t:t[1])[0]
    print(idx)
    return deprocess_image(crops[idx,:,:,:], 224, 224)

showarray(best_image_for_label(crop_scores, 'Egyptian_cat'))
'''
27
'''
```

![image](https://user-images.githubusercontent.com/79494088/139778010-5c40f602-2b88-4762-b5d3-ca64afc2f483.png)

```py
showarray(best_image_for_label(crop_scores, 'Labrador_retriever'))
'''
29
'''
```

![image](https://user-images.githubusercontent.com/79494088/139778088-5ad59c42-4422-4d81-ba55-2bbbb6b3c41c.png)

```py
def create_top_model(base_model):
    inputs = Input(shape=(7, 7, 512), name='input')
    flatten = Flatten(name='flatten')(inputs)
    fc1 = Dense(4096, activation='relu', name='fc1')(flatten)
    fc2 = Dense(4096, activation='relu', name='fc2')(fc1)
    predictions = Dense(1000, activation='softmax', name='predictions')(fc2)
    model = Model(inputs, predictions, name='top_model')
    for layer in model.layers:
        if layer.name != 'input':
            print(layer.name)
            layer.set_weights(base_model.get_layer(layer.name).get_weights())
    return model

top_model = create_top_model(base_model)
top_model.summary()
'''
flatten
fc1
fc2
predictions
Model: "top_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input (InputLayer)           [(None, 7, 7, 512)]       0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000   
=================================================================
Total params: 123,642,856
Trainable params: 123,642,856
Non-trainable params: 0
_________________________________________________________________
'''


bottom_model = vgg16.VGG16(weights='imagenet', include_top=False)
bottom_model.summary()
'''
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
58892288/58889256 [==============================] - 1s 0us/step
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, None, None, 3)]   0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, None, None, 64)    1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, None, None, 64)    36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, None, None, 64)    0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, None, None, 128)   73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, None, None, 128)   147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, None, None, 128)   0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, None, None, 256)   295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, None, None, 256)   590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, None, None, 256)   590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, None, None, 256)   0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, None, None, 512)   1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, None, None, 512)   0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, None, None, 512)   0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________
'''


p0 = base_model.predict(crops[:1])
vgg16.decode_predictions(p0, top=1)
'''
[[('n02099601', 'golden_retriever', 0.40885553)]]
'''


b0 = bottom_model.predict(crops[:1])
t0 = top_model.predict(b0[:, :, :, :])
vgg16.decode_predictions(t0, top=1)
'''
[[('n02099601', 'golden_retriever', 0.40885553)]]
'''
```

- bottom_model을 이용한 output을 crop하여 입력으로 사용한다.
- 이렇게 하면 수행횟수가 64번에서 4번으로 크게 줄어든다.
- 먼저 bottom_model에 이미지를 입력하여 그 결과를 bottom_out에 저장한다.

```py
bottom_out = bottom_model.predict(cat_dog2)
bottom_out.shape
'''
(1, 14, 14, 512)
'''


# bottom_out crop
vec_crops = []
for x in range(7):
    for y in range(7):
        vec_crops.append(bottom_out[0, x: x + 7, y: y + 7, :])
vec_crops = np.asarray(vec_crops)
vec_crops.shape
'''
(49, 7, 7, 512)
'''


t0 = top_model.predict(vec_crops[:1])
vgg16.decode_predictions(t0, top=1)
'''
[[('n02099601', 'golden_retriever', 0.3582282)]]
'''


b0.shape
'''
(1, 7, 7, 512)
'''


crop_pred = top_model.predict(vec_crops)
l = defaultdict(list)
for idx, pred in enumerate(vgg16.decode_predictions(crop_pred, top=1)):
    _, label, weight = pred[0]
    l[label].append((idx, weight))
l.keys()
'''
dict_keys(['golden_retriever', 'tennis_ball', 'tabby', 'tiger_cat', 'Rhodesian_ridgeback', 'Egyptian_cat', 'German_shepherd'])
'''


showarray(best_image_for_label(l, 'golden_retriever'))
'''
35
'''
```

![image](https://user-images.githubusercontent.com/79494088/139779590-4598e99b-d946-45ce-9aba-f8bd4ebb1b9d.png)

```py
showarray(best_image_for_label(l, 'tabby'))
'''
6
'''
```

![image](https://user-images.githubusercontent.com/79494088/139779633-2da15380-0488-4d4b-835a-20f9edf1d41c.png)

# Reference
- [텐서플로우 이미지 분할](https://www.tensorflow.org/tutorials/images/segmentation)
- [github frcnn-from-scratch-with-keras](https://github.com/kentaroy47/frcnn-from-scratch-with-keras)