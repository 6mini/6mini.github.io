---
title: '[Deep Learning] GAN tutorial(QuickDraw 고양이 낙서 생성)'
description: GAN의 대립적인 의미, DCGAN의 Latent 개념와 그 연산, CycleGAN의 개념
categories:
 - Deep Learning
tags: [Deep Learning, GAN, DCGAN, Latent, CycleGAN]
mathjax: enable
---

# QuickDraw Dataset

```py
import json, glob, imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import time


from IPython import display

from tensorflow.keras import layers
from tensorflow.keras.utils import get_file


BASE_PATH = 'https://storage.googleapis.com/quickdraw_dataset/full/binary/'
path = get_file('cat', BASE_PATH + 'cat.bin')


import PIL
from PIL import ImageDraw
from struct import unpack
from sklearn.model_selection import train_test_split

def load_drows(path, train_size=0.85):
    x = []
    # 파일을 풀고 낙서 모으기
    # 낙서는 15바이트 헤더로 시작
    with open(path, 'rb') as f:
        while True:
            img = PIL.Image.new('L', (32, 32), 'white') # 8-bit pixels, black and white #https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
            draw = ImageDraw.Draw(img)
            header = f.read(15)
            if len(header) != 15:
                break
            # 낙서는 x,y 좌표로 구성된 획(stroke) 목록으로 되어 있고, 각 좌표는 분리되어 저장
            # 위에서 생성한 ImageDraw 객체의 좌표 목록을 이용하기 위해 zip()함수를 사용하여 merge
            strokes, = unpack('H', f.read(2))
            for i in range(strokes):
                n_points, = unpack('H', f.read(2))
                fmt = str(n_points) + 'B'
                read_scaled = lambda: (p // 8 for 
                                       p in unpack(fmt, f.read(n_points)))
                points = [*zip(read_scaled(), read_scaled())]               # zip 함수
                draw.line(points, fill=0, width=2)
            img = tf.keras.utils.img_to_array(img)
            x.append(img)
    x = np.asarray(x) / 255
    return train_test_split(x, train_size=train_size)

# 입력받은 10만개의 고양이 낙서 데이터 활용
x_train, x_test = load_drows(path)
print(x_train.shape, x_test.shape) # ((104721, 32, 32, 1), (18481, 32, 32, 1))
```

# GAN

```py
BUFFER_SIZE = 60000
BATCH_SIZE = 256


# 데이터 배치를 만들고 셔플
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256) # 주목: 배치사이즈로 None이 주어진다.

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 32, 32, 1)

    return model


# 생성자를 이용해 이미지 생성
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
```

![image](https://user-images.githubusercontent.com/79494088/140288685-ac938d03-fa99-497f-b74d-94edc42f0505.png)

```py
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[32, 32, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# 감별자를 사용하여 생성된 이미지가 진짜인지 가짜인지 판별
# 진짜 이미지에는 양수값, 가짜 이미지에는 음수값 출력하도록 훈련
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
decision


# 엔트로피 손실함수 (cross entropy loss)를 계산하기 위해 헬퍼 (helper) 함수 반환
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# 감별자와 생성자는 따로 훈련되기 때문에, 감별자와 생성자의 옵티마이저는 다르게 설정
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


EPOCHS = 200
noise_dim = 100
num_examples_to_generate = 16

# 시드 시간이 지나도 재활용
# (GIF 애니메이션에서 진전 내용을 시각화하는데 쉽기 때문) 
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# 이 데코레이터는 함수를 "컴파일"한다.
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # GIF를 위한 이미지 바로 생성
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # 15 에포크가 지날 때마다 모델 저장
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    
    # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # 마지막 에포크가 끝난 후 생성
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)


# 이미지 생성 및 저장
def generate_and_save_images(model, epoch, test_input):
  # `training`이 False로 맞춰진 것 주목
  # (배치정규화를 포함하여) 모든 층들이 추론 모드로 실행
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


%%time
train(train_dataset, EPOCHS)
'''
CPU times: user 32min 37s, sys: 4min 3s, total: 36min 41s
Wall time: 2h 51min 44s
'''
```

![image](https://user-images.githubusercontent.com/79494088/140289976-860cdb07-2655-41d8-93da-2db337ac1e62.png)

- 아니 이거 맞음?ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ
- 200 에포크나 했는데.... 그 누구도 고양이라고 부르지 않을 것 같다...