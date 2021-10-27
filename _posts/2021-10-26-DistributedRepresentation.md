---
title: '[Deep Learning] Distributed Representation'
description: 단어의 분산 표현(원핫인코딩의 단점, 분포 기반의 표현, 임베딩), Word2Vec(CBoW, Skip-gram 차이), fastText(OOV문제, 철자 단위 임베딩 방법의 장점)
categories:
 - Deep Learning
tags: [Deep Learning, 원핫인코딩, Embedding, 임베딩, Word2Vec, CBoW, Skip-gram, fastText, OOV, 철자 단위 임베딩]
mathjax: enable
---

# Warm Up
- 분포 표현(Distributed Representation): 단어 분포를 중심으로 단어를 벡터화한다.
- 대표적 방법 중 하나는 Word2Vec(워드투벡터)이다.
- [[딥러닝 자연어처리]Word2Vec](https://youtu.be/sY4YyacSsLc)
    - 원핫인코딩은 유사도가 없다.
    - Embedding
        - 유사도를 구할 수 있다.
        - 저차원이다.

![image](https://user-images.githubusercontent.com/79494088/138715004-49783f84-f7b6-46f1-9b17-0151f27e9768.png)

# Distributed Representation
- 단어 자체를 벡터화하는 방법이다.
- Word2Vec, fastText는 **벡터로 표현하고자 하는 타겟 단어(Target word)가 해당 단어 주변 단어에 의해 결정**된다.
- 단어 벡터를 이렇게 정하는 이유는 [분포 가설(Distribution hypothesis)](https://en.wikipedia.org/wiki/Distributional_semantics) 때문이다.
    - **'비슷한 위치에서 등장하는 단어는 비슷한 의미를 가진다'**
    - 비슷한 의미를 지닌 단어는 주변 단어 분포도 비슷하다.
- **분포 가설에 기반하여 주변 단어 분포를 기준으로 단어의 벡터 표현이 결정되기 때문에 분산 표현(Distirbuted representation)**이라고 부른다.

## One hot Encoding
- 단어를 벡터화하고자 할 때 선택할 수 있는 쉬운 방법이다.
- 치명적인 단점이 단어 간 유사도를 구할 수 없다는 점이다.
- 단어 간 유사도를 구할 때 코사인 유사도가 자주 사용된다.
- 코사인 유사도:

$$
\text{Cosine similarity} = \frac{\vec{a} \cdot \vec{b} }{\vert \vec{a} \vert \vert \vec{b} \vert }
$$

- 원핫인코딩을 사용한 두 벡터의 내적은 항상 0이므로 어떤 두 단어를 골라 코사인 유사도를 구하더라도 그 값은 0이 된다.
- 이 때문에 어떤 단어를 골라도 두 단어 사이의 관계를 전혀 알 수 없게 된다.

## Embedding
- 원핫인코딩의 단점을 해결하기 위해 등장했다.
- 단어를 고정 길이의 벡터, 즉 차원이 일정한 벡터로 나타내기 때문에 'Embedding'이라는 이름이 붙었다.
- 벡터 내의 각 요소가 연속적인 값을 가지게 된다.
- 가장 널리 알려진 방법은 Word2Vec이다.

# Word2Vec
- **단어를 벡터로 나타내는 방법**으로 가장 많이 사용된다.
- **특정 단어 양 옆에 있는 두 단어의 관계를 활용**하기 때문에 분포 가설을 잘 반영하고 있다.
- CBoW와 Skip-gram의 2가지 방법이 있다.

## CBoW & Skip-gram
- 차이
    - 주변 단어에 대한 정보를 기반으로 중심 단어의 정보를 예측하는 모델 ▶️ **<font color="ff6f61">CBoW(Continuous Bag-of-Words)</font>**
    - 중심 단어의 정보를 기반으로 주변 단어의 정보를 예측하는 모델 ▶️ **<font color="ff6f61">Skip-gram</font>**

![image](https://user-images.githubusercontent.com/79494088/138725010-52f01e53-44f7-497b-b629-5af78b323ecc.png)

- 예시
    - <별 헤는 밤>의 일부분에 형태소 분석기 적용하여 토큰화

> “… 어머님 나 는 별 하나 에 아름다운 말 한마디 씩 불러 봅니다 …”

**CBoW**를 사용하면 표시된 단어 정보를 바탕으로 아래의 [ ---- ] 에 들어갈 단어를 예측하는 과정으로 학습이 진행된다.

> “… 나 는 [ -- ] 하나 에 … “ <br/>
> “… 는 별 [ ---- ] 에 아름다운 …”<br/>
> “… 별 하나 [ -- ] 아름다운 말 …”<br/>
> “… 하나 에 [ -------- ] 말 한마디 …”

**Skip-gram**을 사용하면 표시된 단어 정보를 바탕으로 다음의 [ ---- ] 에 들어갈 단어를 예측하는 과정으로 학습이 진행된다.

> “… [ -- ] [ -- ] 별 [ ---- ] [ -- ] …” <br/>
> “… [ -- ] [ -- ] 하나 [ -- ] [ -------- ] …” <br/>
> “… [ -- ] [ ---- ] 에 [ -------- ] [ -- ] …” <br/>
> “… [ ---- ] [ -- ] 아름다운 [ -- ] [ ------ ] …”

- 더 많은 정보를 바탕으로 특정 단어를 예측하기 때문에 CBoW의 성능이 더 좋을 것으로 생각할 수 있지만, 역전파 관점에서 보면 Skip-gram에서 훨씬 더 많은 학습이 일어나기 때문에 Skip-gram의 성능이 조금 더 좋게 나타난다.(물론 계산량이 많기 때문에 리소스도 더 크다.)

## 구조
- Skip-gram 기준 Word2Vec 구조
    - 입력: Word2Vec의 입력은 원핫인코딩된 단어 벡터
    - 은닉: 임베딩 벡터의 차원수 만큼의 노드로 구성된 은닉층이 1개인 신경망
    - 출력: 단어 개수 만큼의 노드로 이루어져 있으며 활성화 함수로 소프트맥스를 사용
- 논문에서는 총 10,000개의 단어에 대해서 300차원의 임베딩 벡터를 구했기 때문에 신경망 구조가 아래와 같다.

<img src="http://mccormickml.com/assets/word2vec/skip_gram_net_arch.png" width="800" />

## 학습 데이터 디자인
- 효율적인 Word2Vec 학습을 위해 학습데이터를 잘 구성해야 한다.
- Window 사이즈가 2인 Word2Vec 이므로 중심 단어 옆에 있는 2개 단어에 대해 단어쌍을 구성한다.
- 만약 '**"The tortoise jumped into the lake"** 라는 문장에 대해 단어쌍을 구성한다면, 윈도우 크기가 2인 경우 다음과 같이 Skip-gram을 학습하기 위한 데이터 쌍을 구축할 수 있다.
    - 중심 단어 : **The**, 주변 문맥 단어 : tortoise, jumped
        - 학습 샘플: (the, tortoise), (the, jumped)
    - 중심 단어 : **tortoise**, 주변 문맥 단어 : the, jumped, into
        - 학습 샘플: (tortoise, the), (tortoise, jumped), (tortoise, into)
    - 중심 단어 : **jumped**, 주변 문맥 단어 : the, tortoise, into, the
        - 학습 샘플: (jumped, the), (jumped, tortoise), (jumped, into), (jumped, the)
    - 중심 단어 : **into**, 주변 문맥 단어 : tortoise, jumped, the, lake
        - 학습 샘플: (into, tortoise), (into, jumped), (into, the), (into, lake)
- 다음과 같은 데이터쌍이 만들어 진다.

|중심단어|문맥단어|
|the|tortoise|
|the|jumped|
|tortoise|the|
|tortoise|jumped|
|tortoise|into|
|jumped|the|
|jumped|tortoise|
|jumped|into|
|jumped|the|
|into|tortoise|
|into|jumped|
|into|the|
|into|lake|
|...|...|

## 결과
- 학습이 모두 끝나면 10000개의 단어에 대해 300차원의 임베딩 벡터가 생성된다.
- 임베딩 벡터의 차원을 조절하고 싶다면 은닉층의 노드 수를 줄이거나 늘릴 수 있다.
- 아래 그림은 신경망 내부에 있는 10000*300 크기의 가중치 행렬에 의해 10000개 단어에 대한 300 차원의 벡터가 생성되는 모습을 나타낸 이미지이다.

![image](https://user-images.githubusercontent.com/79494088/138727169-6ae1fea8-f7ce-4d7d-9def-b9565f735433.png)

### 효율을 높이기 위한 기법

#### Nagative-sampling

$$ p(w_i) = \frac{f(w_i)^{3/4}}{\sum ^{10000}_{j=1} f(w_j)^{3/4}} $$

- Nagative 값을 샘플링하는 것이다.
- 무관한 단어들에 대해 weight를 업데이트하지 않아도 된다.
- n개의 negative 값을 선택하고 이 값들에 대해서만 positive 값과 함게 학습한다.
- 논문에서 negative 갯수인 n은 작은 데이터에서는 5-20, 큰 데이터에서는 2-5라고 제시하였다.

#### Subsampling

$$ P(w_i) (\sqrt{ \frac{z(w_i)}{0.001}}+1) \frac{0.001}{z(w_i)} $$

- 텍스트 자체가 가진 문제를 다룬 방법이다.
- 자주 등장하지만 별 쓸모는 없는 단어를 다룬다.


- 결과적으로 Skip-gram 모델을 통해 10000개의 단어에 대한 임베딩 벡터를 얻었다.
- 이렇게 얻은 임베딩 벡터는 문장 간 관련도 계산, 문서 분류같은 작업에 사용할 수 있다.

## 임베딩 벡터 시각화
- 임베딩 벡터는 단어간의 의미적, 문법적 관계를 잘 나타낸다.
- **`man - woman`** 사이의 관계와 **`king - queen`** 사이의 관계가 매우 유사하다.<br>
생성된 임베딩 벡터가 단어의 **의미적(Semantic) 관계를 잘 표현**한다.

- **`walking - walked`** 사이의 관계와 **`swimming - swam`** 사이의 관계가 매우 유사하다.<br>
생성된 임베딩 벡터가 단어의 **문법적(혹은 구조적, Syntactic)인 관계도 잘 표현**한다.

- 고유명사에 대해서도 나라 - 수도 와 같은 관계를 잘 나타내고 있다. 

## 실습

### gensim 패키지

```py
# 업그레이드
!pip install gensim --upgrade


# 구글 뉴스 말뭉치로 학습된 word2vec 벡터 다운
import gensim.downloader as api

wv = api.load('word2vec-google-news-300')


# 단어 확인
for idx, word in enumerate(wv.index_to_key):
    if idx == 10:
        break

    print(f"word #{idx}/{len(wv.index_to_key)} is '{word}'")
'''
word #0/3000000 is '</s>'
word #1/3000000 is 'in'
word #2/3000000 is 'for'
word #3/3000000 is 'that'
word #4/3000000 is 'is'
word #5/3000000 is 'on'
word #6/3000000 is '##'
word #7/3000000 is 'The'
word #8/3000000 is 'with'
word #9/3000000 is 'said'
'''


# 임베딩 벡터의 차원과 값
# king이라는 단어의 벡터의 shape을 출력하여 임베딩 벡터의 차원 확인
# 임베딩 벡터는 300차원이며, 벡터의 요소가 원핫인코딩과 다르다.
vec_king = wv['king']

print(f"Embedding dimesion is : {vec_king.shape}\n")
print(f"Embedding vector of 'king' is \n\n {vec_king}")
'''
Embedding dimesion is : (300,)

Embedding vector of 'king' is 

 [ 1.25976562e-01  2.97851562e-02  8.60595703e-03  1.39648438e-01
 -2.56347656e-02 -3.61328125e-02  1.11816406e-01 -1.98242188e-01
  5.12695312e-02  3.63281250e-01 -2.42187500e-01 -3.02734375e-01
 '
 '
 '
  2.99072266e-02 -5.93261719e-02 -4.66918945e-03 -2.44140625e-01
 -2.09960938e-01 -2.87109375e-01 -4.54101562e-02 -1.77734375e-01
 -2.79296875e-01 -8.59375000e-02  9.13085938e-02  2.51953125e-01]
'''


# 단어 간 유사도 파악
# .similarity 활용
pairs = [
    ('car', 'minivan'),   
    ('car', 'bicycle'),  
    ('car', 'airplane'),
    ('car', 'cereal'),    
    ('car', 'democracy')
]

for w1, w2 in pairs:
    print(f'{w1} ======= {w2}\t  {wv.similarity(w1, w2):.2f}')
'''
car ======= minivan	  0.69
car ======= bicycle	  0.54
car ======= airplane  0.42
car ======= cereal	  0.14
car ======= democracy 0.08
'''


# car 벡터에 minivan 벡터를 더한 벡터와 가장 유사한 5개의 단어
# .most_similar 메소드 사용
for i, (word, similarity) in enumerate(wv.most_similar(positive=['car', 'minivan'], topn=5)):
    print(f"Top {i+1} : {word}, {similarity}")
'''
Top 1 : SUV, 0.8532192707061768
Top 2 : vehicle, 0.8175783753395081
Top 3 : pickup_truck, 0.7763688564300537
Top 4 : Jeep, 0.7567334175109863
Top 5 : Ford_Explorer, 0.7565720081329346
'''


# king 벡터에 women 벡터를 더한 뒤 men 벡터를 빼주면 queen이 나온다.
# Walking 벡터에 swam 벡터를 더한 뒤 walked 벡터를 빼주면 swimming이 나온다.
print(wv.most_similar(positive=['king', 'women'], negative=['men'], topn=1))
print(wv.most_similar(positive=['walking', 'swam'], negative=['walked'], topn=1))
'''
[('queen', 0.6525818109512329)]
[('swimming', 0.7448815703392029)]
'''


# 가장 관계 없는 단어 뽑기
# .doesnt_match 메소드 사용
print(wv.doesnt_match(['fire', 'water', 'land', 'sea', 'air', 'car']))
'''
car
'''
```

# fastText
- fastText는 Word2Vec 방식에 철자(Character) 기반의 임베딩 방식을 더해준 새로운 임베딩 방식이다.

## OOV(Out of Vocabulary) 문제
- 세상 모든 단어가 들어있는 말뭉치를 구하는 것은 불가능하다.
- Word2Vec은 말뭉치에 등장하지 않은 단어에 대해 임베딩 벡터를 만들지 못한다는 단점이 있다.
- **기존 말뭉치에 등장하지 않는 단어가 등장하는 문제를 OOV 문제**라고 한다.
- 또한 적게 등장하는 단어에 대해 학습이 적게 일어나기 때문에 적절한 임베딩 벡터를 생성해내지 못한다는 것도 Word2Vec의 단점이다.

## 철자 단위 임베딩(Character level Embedding)
- fastText는 철자(Character) 수준의 임베딩을 보조 정보로 사용함으로써 OOV 문제를 해결했다.
- 모델이 학습하지 못한 단어더라도 잘 쪼개고 보면 말뭉치에서 등장했던 단어를 통해 유추해 볼 수 있다는 아이디어에서 출발했다.
- **fastText가 Character-level(철자 단위) 임베딩을 적용하는 법: <font color='red'>Character n-gram</font>**
- 3-6개로 묶은 Character 정보(3-6 grams) 단위를 사용한다.
- 묶기 이전에 모델이 접두사와 접미사를 인식할 수 있도록 해당 단어 앞뒤로 <, >를 붙여준다.

![image](https://user-images.githubusercontent.com/79494088/138729948-2f2e2aa7-6406-48d7-8e37-f3878b32167d.png)

| word   | Length(n) | Character n-grams            |
| eating | 3         | <ea, eat, ati, tin, ing, ng> |
| eating | 4         | <eat, eati, atin, ting, ing> |
| eating | 5         | <eati, eatin, ating, ting>   |
| eating | 6         | <eatin, eating, ating>       |

- 총 18개의 Character-level n-gram을 얻을 수 있다.
- 알고리즘이 매우 효율적으로 구성되어 있기 때문에 시간상으로 Word2Vec과 엄청난 차이가 있지는 않다.

### 적용
    - eating이라는 단어가 말뭉치 내에 있다면 skip-gram으로부터 학습된 임베딩 벡터에 위에서 얻은 18개 Character-level n-gram의 벡터를 더해준다.
    - 반대로, eating이라는 단어가 말뭉치에 없다면 18개 Character-level n-gram의 벡터만으로 구성한다.

### 시각화

![image](https://user-images.githubusercontent.com/79494088/138730977-94c1f11b-5dfc-4f61-bff6-5710bf944aae.png)

![image](https://user-images.githubusercontent.com/79494088/138731034-dd3628e9-7dec-4b02-a32c-5018c11b5175.png)

### 실습

#### gensim 패키지

```py
from pprint import pprint as print
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath

# Set file names for train and test data
corpus_file = datapath('lee_background.cor')

model = FastText(vector_size=100)

# build the vocabulary
model.build_vocab(corpus_file=corpus_file)

# train the model
model.train(
    corpus_file=corpus_file, epochs=model.epochs,
    total_examples=model.corpus_count, total_words=model.corpus_total_words,
)

print(model)


# night, nights 각각 사전에 있는 지 확인
ft = model.wv
print(ft)

print(f"night => {'night' in ft.key_to_index}")
print(f"nights => {'nights' in ft.key_to_index}")
'''
'night => True'
'nights => False'
'''


# 임베딩 벡터 확인
print(ft['night'])
'''
array([-1.21940151e-01,  9.35477093e-02, -2.68753201e-01, -9.21401829e-02,
        5.67255244e-02,  3.27864051e-01,  3.91383469e-01,  5.69616437e-01,
        1.93194106e-01, -2.93112427e-01,  6.31607324e-02, -1.48656189e-01,
       -2.79613197e-01,  5.90286553e-01, -3.61445814e-01, -5.47924638e-01,
        1.34900540e-01, -2.14206606e-01, -4.45417851e-01, -5.28838873e-01,
       -4.67930526e-01,  5.05698696e-02, -5.71829677e-01, -1.30317435e-01,
       -1.92824587e-01, -2.69073665e-01, -5.83209455e-01, -1.01116806e-01,
       -2.19189227e-01,  1.81627348e-01, -2.94398159e-01,  2.68787891e-01,
        8.11280549e-01, -2.19889328e-01,  2.07663789e-01,  2.78767705e-01,
        4.31476295e-01, -2.91228201e-02, -3.44264716e-01, -2.94634223e-01,
        5.33911526e-01, -4.73139107e-01,  1.09619908e-01, -3.26181561e-01,
       -5.31040490e-01, -3.78617615e-01,  3.16744496e-04,  1.84120595e-01,
        2.58488834e-01, -2.17340793e-02,  3.49597663e-01, -5.01383007e-01,
        2.72206813e-01, -4.29527640e-01, -2.38943994e-01, -2.56465763e-01,
       -2.04503626e-01, -1.21376544e-01,  2.89444923e-02, -3.19146156e-01,
       -3.51654828e-01, -4.28357989e-01, -3.23994905e-01,  3.59695762e-01,
       -7.61662573e-02,  7.05648482e-01,  4.38806303e-02,  1.43223805e-02,
        3.75390232e-01,  3.57252628e-01, -2.17079744e-01,  4.64041203e-01,
        5.86148262e-01, -6.49744213e-01,  2.93909192e-01, -3.06044258e-02,
        2.57350951e-01, -7.78491274e-02,  8.48653764e-02,  3.41358751e-01,
        1.31611019e-01, -4.65880662e-01, -7.48033941e-01, -1.79655612e-01,
       -1.04323834e-01, -7.16210783e-01,  4.64819491e-01,  2.32284784e-01,
       -3.15516964e-02, -3.45896810e-01, -8.36842135e-02,  3.75827789e-01,
       -1.90781638e-01,  6.69103414e-02, -2.75694251e-01,  5.48373401e-01,
       -1.67792484e-01, -2.27489457e-01,  1.73320379e-02, -2.78268337e-01],
      dtype=float32)
'''


print(ft['nights'])
'''
array([-0.1060066 ,  0.08164798, -0.23207934, -0.07934359,  0.04771318,
        0.28185087,  0.33920357,  0.49367297,  0.16726466, -0.25471696,
        0.05623917, -0.12683469, -0.24250235,  0.50755054, -0.3133934 ,
       -0.47360393,  0.11576562, -0.18443272, -0.38318864, -0.45724452,
       -0.4011069 ,  0.04254517, -0.49356073, -0.11393045, -0.16512693,
       -0.23088636, -0.5021323 , -0.08505005, -0.18894437,  0.15834409,
       -0.25215933,  0.23156057,  0.6985802 , -0.1893784 ,  0.17935547,
        0.24007267,  0.37383503, -0.02515006, -0.29740998, -0.25486094,
        0.45998394, -0.40748954,  0.09404191, -0.2810496 , -0.4591848 ,
       -0.32557708,  0.0031702 ,  0.15951274,  0.22447531, -0.01768246,
        0.30329126, -0.4329943 ,  0.2356085 , -0.37062383, -0.20580977,
       -0.22011346, -0.17842367, -0.10299528,  0.02632287, -0.27278537,
       -0.3024929 , -0.3700565 , -0.2792666 ,  0.30997086, -0.06514771,
        0.61006665,  0.03797323,  0.00971325,  0.32373375,  0.3095027 ,
       -0.18784912,  0.39859974,  0.50703657, -0.5606495 ,  0.2551437 ,
       -0.02540389,  0.22150257, -0.06802121,  0.07301091,  0.2946853 ,
        0.11406258, -0.40247047, -0.64501   , -0.15606925, -0.08905818,
       -0.6193155 ,  0.40114415,  0.20066874, -0.02517485, -0.29917863,
       -0.07218475,  0.32315075, -0.16517399,  0.05838989, -0.23842031,
        0.47345078, -0.14618942, -0.19270451,  0.01530515, -0.24100253],
      dtype=float32)
'''


# 두 단어 유사도 확인
print(ft.similarity("night", "nights"))
'''
0.9999918
'''


# nights와 가장 비슷한 단어
print(ft.most_similar("nights"))
'''
[('night', 0.9999917149543762),
 ('rights', 0.9999875426292419),
 ('flights', 0.9999871850013733),
 ('overnight', 0.9999868273735046),
 ('fighters', 0.9999852776527405),
 ('fighting', 0.9999851584434509),
 ('entered', 0.9999849796295166),
 ('fight', 0.999984860420227),
 ('fighter', 0.9999845027923584),
 ('night.', 0.9999843835830688)]
'''
# 주로 비슷하게 생긴 단어가 많이 속해있다.


# 가장 관련 없는 단어
print(ft.doesnt_match("night noon fight morning".split()))
'''
'noon'
'''
```

- 단어 뜻만 보면 fight가 나와야 하지만, 뜬금없기 noon이 등장했다.
- **fastText 임베딩 벡터는 단어의 의미보다 결과 쪽에 조금 더 비중을 두고 있다.**

# 문장 분류 수행
- 가장 간단한 것은 문장에 있는 단어 벡터를 모두 더한 후 평균내어 구하는 방법이다.
- 간단한 문제에 대해 좋은 성능을 보여서 baseline 모델로 많이 사용된다.

```py
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.datasets import imdb


tf.random.set_seed(42)


(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=20000)


X_train[0]
'''
[1,
 14,
 22,
'
'
'
 19,
 178,
 32]
'''


# 인덱스로 된 데이터 텍스트로 변경하는 함수 구현
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    """
    word_index를 받아 text를 sequence 형태로 반환하는 함수
    """
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


decode_review(X_train[0])
'''
the as you with out themselves powerful lets loves their becomes reaching had journalist of lot from anyone to have after out atmosphere never more room and it so heart shows to years of every never going and help moments or of every chest visual movie except her was several of enough more with is now current film as you of mine potentially unfortunately of you than him that with out themselves her get for was camp of you movie sometimes movie that with scary but pratfalls to story wonderful that in seeing in character to of 70s musicians with heart had shadows they of here that with her serious to have does when from why what have critics they is you that isn't one will very to as itself with other tricky in of seen over landed for anyone of and br show's to whether from than out themselves history he name half some br of 'n odd was two most of mean for 1 any an boat she he should is thought frog but of script you not while history he heart to real at barrel but when from one bit then
'''


# keras tokenizer에 텍스트 학습
sentences = [decode_review(idx) for idx in X_train]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)


vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
'''
19999
'''


# pad_sequence를 통해 패딩 처리(찾아보기)
X_encoded = tokenizer.texts_to_sequences(sentences)

max_len = max(len(sent) for sent in X_encoded)
print(max_len)
'''
2494
'''


print(f'Mean length of train set: {np.mean([len(sent) for sent in X_train], dtype=int)}')
'''
Mean length of train set: 238
'''


X_train=pad_sequences(X_encoded, maxlen=400, padding='post')
y_train=np.array(y_train)


# word2vec의 임베딩 가중치 행렬 생성
# 미리 학습된 모든 단어에 대해 만들 경우 너무 행렬이 커지므로 vocab에 속하는 단어에 대해서만 만들어지도록 한다.
embedding_matrix = np.zeros((vocab_size, 300))

print(np.shape(embedding_matrix))
'''
(19999, 300)
'''


def get_vector(word):
    """
    해당 word가 word2vec에 있는 단어일 경우 임베딩 벡터를 반환
    """
    if word in wv:
        return wv[word]
    else:
        return None
 
for word, i in tokenizer.word_index.items():
    temp = get_vector(word)
    if temp is not None:
        embedding_matrix[i] = temp


# 신경망을 구성하기 위한 keras 모듈을 불러온 후 학습을 수행
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten


model = Sequential()
model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(GlobalAveragePooling1D()) # 입력되는 단어 벡터의 평균을 구하는 함수
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y_train, batch_size=64, epochs=20, validation_split=0.2)
'''
Epoch 1/20
WARNING:tensorflow:Model was constructed with shape (None, 2494) for input KerasTensor(type_spec=TensorSpec(shape=(None, 2494), dtype=tf.float32, name='embedding_input'), name='embedding_input', description="created by layer 'embedding_input'"), but it was called on an input with incompatible shape (None, 400).
WARNING:tensorflow:Model was constructed with shape (None, 2494) for input KerasTensor(type_spec=TensorSpec(shape=(None, 2494), dtype=tf.float32, name='embedding_input'), name='embedding_input', description="created by layer 'embedding_input'"), but it was called on an input with incompatible shape (None, 400).
311/313 [============================>.] - ETA: 0s - loss: 0.6924 - acc: 0.5268WARNING:tensorflow:Model was constructed with shape (None, 2494) for input KerasTensor(type_spec=TensorSpec(shape=(None, 2494), dtype=tf.float32, name='embedding_input'), name='embedding_input', description="created by layer 'embedding_input'"), but it was called on an input with incompatible shape (None, 400).
313/313 [==============================] - 8s 24ms/step - loss: 0.6924 - acc: 0.5268 - val_loss: 0.6907 - val_acc: 0.5996
Epoch 2/20
313/313 [==============================] - 6s 20ms/step - loss: 0.6901 - acc: 0.5767 - val_loss: 0.6882 - val_acc: 0.5972
Epoch 3/20
313/313 [==============================] - 6s 20ms/step - loss: 0.6880 - acc: 0.5899 - val_loss: 0.6859 - val_acc: 0.6006
Epoch 4/20
313/313 [==============================] - 7s 23ms/step - loss: 0.6861 - acc: 0.5918 - val_loss: 0.6837 - val_acc: 0.5928
Epoch 5/20
313/313 [==============================] - 7s 23ms/step - loss: 0.6845 - acc: 0.5949 - val_loss: 0.6824 - val_acc: 0.5970
Epoch 6/20
313/313 [==============================] - 6s 20ms/step - loss: 0.6827 - acc: 0.5957 - val_loss: 0.6800 - val_acc: 0.5946
Epoch 7/20
313/313 [==============================] - 7s 22ms/step - loss: 0.6813 - acc: 0.6028 - val_loss: 0.6787 - val_acc: 0.6112
Epoch 8/20
313/313 [==============================] - 6s 20ms/step - loss: 0.6798 - acc: 0.6058 - val_loss: 0.6774 - val_acc: 0.6092
Epoch 9/20
313/313 [==============================] - 7s 22ms/step - loss: 0.6784 - acc: 0.6048 - val_loss: 0.6755 - val_acc: 0.6150
Epoch 10/20
313/313 [==============================] - 7s 21ms/step - loss: 0.6772 - acc: 0.6078 - val_loss: 0.6742 - val_acc: 0.6168
Epoch 11/20
313/313 [==============================] - 7s 21ms/step - loss: 0.6759 - acc: 0.6100 - val_loss: 0.6730 - val_acc: 0.6172
Epoch 12/20
313/313 [==============================] - 7s 22ms/step - loss: 0.6748 - acc: 0.6116 - val_loss: 0.6716 - val_acc: 0.6192
Epoch 13/20
313/313 [==============================] - 7s 22ms/step - loss: 0.6736 - acc: 0.6095 - val_loss: 0.6712 - val_acc: 0.6128
Epoch 14/20
313/313 [==============================] - 7s 21ms/step - loss: 0.6725 - acc: 0.6135 - val_loss: 0.6690 - val_acc: 0.6246
Epoch 15/20
313/313 [==============================] - 6s 19ms/step - loss: 0.6712 - acc: 0.6158 - val_loss: 0.6679 - val_acc: 0.6212
Epoch 16/20
313/313 [==============================] - 6s 20ms/step - loss: 0.6704 - acc: 0.6147 - val_loss: 0.6673 - val_acc: 0.6222
Epoch 17/20
313/313 [==============================] - 6s 19ms/step - loss: 0.6693 - acc: 0.6191 - val_loss: 0.6670 - val_acc: 0.6196
Epoch 18/20
313/313 [==============================] - 6s 20ms/step - loss: 0.6684 - acc: 0.6217 - val_loss: 0.6649 - val_acc: 0.6268
Epoch 19/20
313/313 [==============================] - 7s 21ms/step - loss: 0.6674 - acc: 0.6205 - val_loss: 0.6641 - val_acc: 0.6266
Epoch 20/20
313/313 [==============================] - 7s 21ms/step - loss: 0.6666 - acc: 0.6248 - val_loss: 0.6633 - val_acc: 0.6274
<tensorflow.python.keras.callbacks.History at 0x7f3db9f0ef10>
'''


test_sentences = [decode_review(idx) for idx in X_test]

X_test_encoded = tokenizer.texts_to_sequences(test_sentences)

X_test=pad_sequences(X_test_encoded, maxlen=400, padding='post')
y_test=np.array(y_test)


model.evaluate(X_test, y_test)
'''
782/782 [==============================] - 7s 8ms/step - loss: 0.6680 - acc: 0.6093
[0.6679654121398926, 0.609279990196228]
'''
```

# Review
- 원-핫 인코딩이란?
- 원-핫 인코딩 단점은?
- 분포 기반의 표현, 임베딩이란?
- 분포 가설이란?
- 임베딩 벡터는 특징은?
- CBoW와 Skip-gram의 차이는 무엇이며, 어떤 방법이 성능이 더 좋을까? 성능이 더 좋은 방법의 단점은?
- Word2Vec의 임베딩 벡터를 시각화한 결과가 어떤 특징을 가지는지?
- Word2Vec의 단점은?
- OOV 문제란?
- 철자(Character) 단위 임베딩 방법의 장점은?

## 분포 가설(Distributed Hypothesis)
- 비슷한 위치에서 등장하는 단어는 비슷한 의미를 가진다.
## Word2Vec
- 원핫인코딩은 단어 간 유사도를 계산을 할 수 없다는 단점이 있는데, 유사도를 반영할 수 있도록 단어의 의미를 벡터화 하는 방법이다.
- 주변에 있는 단어로 중간에 있는 단어를 예측하는 CBoW와 중간에 있는 단어로 주변에 있는 단어를 예측하는 Skip-Gram 방식이 있다.
- 모르는 단어(OOV)는 분석할 수 없다.
## fastText
- Word2Vec은 단어를 쪼갤 수 없는 단위로 생각한다면, festText는 하나의 단어 안에도 여러 단어가 존재하는 것으로 간주한다.
- 글자 단위의 n-gram의 구성으로 취급하며, n의 수에 따라 단어가 얼마나 분리되는지 결정된다.
- 모르는 단어(OOV)도 분석할 수 있다.
## GloVe
- 임베딩된 두 단어벡터의 내적이 말뭉치 전체에서의 동시 등장확률 로그값이 되도록 목적함수를 정의하여 임베딩 된 단어벡터 간 유사도 측정을 수월하게 하면서도 말뭉치 전체의 통계정보를 반영하기위해 나온 방법이다.

# References
- [n-gram](https://www.youtube.com/watch?v=4f9XC8HHluE)
- [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)