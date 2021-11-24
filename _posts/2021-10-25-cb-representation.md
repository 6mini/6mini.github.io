---
title: '[Deep Learning] Count Based Representation'
description: 자연어와 전처리(SpaCy 라이브러리 활용 토큰화, 불용어, 어간 및 표제어 추출), 등장 횟수 기반의 단어 표현(문서-단어 행렬, BoW, TF-IDF)
categories:
 - Deep Learning
tags: [Deep Learning, NLP, SpaCy, 불용어, Stop words, 어간 추출, 표제어 추출, Stemming, Lemmatization, 등장 횟수 기반 단어 표현, Count Based Representation, BoW, TF-IDF]
mathjax: enable
---

## 자연어 처리 용어
- Corpus(말뭉치): 특정한 목적을 가지고 수집한 텍스트 데이터
- Document(문서): Sentence(문장)의 집합
- Sentence: 여러개의 토큰(단어, 형태소)로 구성된 문자열. 마침표, 느낌표 등의 기호로 구분
- Vocabulary(어휘집합): 코퍼스에 있는 모든 문서, 문장을 토크화한 후 중복을 제거한 토큰의 집합
- 전처리(Preprocessing)
    - 토큰화(Tokenization)
    - 차원의 저주(Curse of Dimensionality)
    - 불용어(Stop words)
    - 어간 추출(Stemming)
    - 표제어 추출(Lemmatization)

## 등장 횟수 기반의 단어 표현(Count-based Representation)

### [민트 초코 논란! 자연어 처리(NLP)로 종결해드림](https://www.youtube.com/watch?v=QTgRNer2epE)
- 민트초코 및 하와이안피자 호불호 비율 7:3으로 '호' 압승

### [Bag of Words](https://www.youtube.com/watch?v=dKYFfUtij_U)
- 횟수 기반의 단어
- Bag of Words: 문장을 숫자로 표현하는 방법 중 하나
    - Sentence 혹은 Document를 벡터로 나타내는 가장 단순한 방법

![image](https://user-images.githubusercontent.com/79494088/138591580-ff155e7d-0040-4a68-bc4c-41456fbcfe43.png)

- 활용
    - 문장의 유사도 확인
    - 머신러닝 모델 입력값으로 사용
- 단점
    - Sparsity: 실제 사전에는 백 만개가 넘는 단어가 있다.
        - 계산량이 높아지고 메모리가 많이 사용된다.
    - 많이 출현한 단어는 힘이 세진다.
    - 단어의 순서를 철저히 무시한다.
    - 처음 보는 단어는 처리하지 못한다(오타, 줄임말).

### [TF-IDF](https://www.youtube.com/watch?v=meEchvkdB1U)
- Term Frequency-Inverse Document Frequency
- 각 단어별로 연관성을 알고싶을 때 사용되며, 수치로 표현된다.
- TF: 문서가 주어졌을 때 단어가 얼마나 출현했는지의 값

![image](https://user-images.githubusercontent.com/79494088/138628962-128516f2-9796-4c08-8390-3275ed8b012c.png)

- IDF: 자주 출현하는 단어에 대해 패널티를 준다.

![image](https://user-images.githubusercontent.com/79494088/138629144-1f11477a-288e-45a0-a0cc-8c2b7936e20b.png)

# 자연어 처리

## 자연어
- Netural Language: **사람이 일상적으로 쓰는 언어를 인공적으로 만들어진 언어인 인공어와 구분하여 부르는 개념**
- 자연어가 아닌 것: 에스페란토어, 코딩 언어 등
- NLP(Natural Language Processing, 자연어 처리): 자연어를 컴퓨터로 처리하는 기술
- Text Mining을 잘하는 방법

![image](https://user-images.githubusercontent.com/79494088/138671125-8ec26db0-9d3e-4cda-82b3-8d03608620bd.png)


## 자연어 처리로 할 수 있는 일

### 자연어 이해(NLU, Natural Language Understanding)
- 분류: 뉴스 기사 분류, 감정 분석
- 자연어 추론(NLI, Natural Langauge Inference)
    - ex) A는 B에게 암살당했다 -> A는 죽었다 -> T or F
- 기계 독해(MRC, Machine Reading Comprehension), 질의 응답(QA, Question&Answering)
- 품사 태깅(POS tagging), 개체명 인식(Named Entity Recognition) 등
    - [POS Tagging](https://excelsior-cjh.tistory.com/71)
    - [NER](https://stellarway.tistory.com/29)

### 자연어 생성(NLG, Natural Language Generation)
- 텍스트 생성
    ex) 뉴스 기사 생성, 가사 생성

### NLU & NLG
- 기계 번역(Machine Translation)
- 요약(Summerization)
    - 추출 요약(Extractive summerization): 문서 내에서 해당 문서를 가장 잘 요약하는 부분을 찾아내는 Task -> NLU에 가깝다.
    - 생성 요약(Absractive summerization): 해당 문서를 요약하는 요약문 생성 -> NLG에 가깝다.
- 챗봇(Chatbot)
    - 특정 테스크를 처리하기 위한 챗봇(Task Oriented Dialog, TOD)
        - ex) 식당 예약을 위한 챗봇, 상담 응대를 위한 챗봇
    - 정해지지 않은 주제를 다루는 일반 대화 챗봇(Open Domain Dialog, ODD)

### 기타
- TTS(Text to Speech): 텍스트를 음성으로 읽기(ex) 슈퍼챗)
- STT(Speech to Text): 음성을 텍스트로 쓰기(ex) 컨퍼런스, 강연 등에서 청각 장애인을 위한 실시간 자막 서비스)
- Image Captioning: 이미지를 설명하는 문장 생성

## 사례
- 챗봇: 심리상담 챗봇(트로스트, 아토머스), 일반대화 챗봇(스캐터랩 - 이루다, 마인드로직)
- 번역: 파파고, 구글 번역기
- TTS, STT: 인공지능 스피커, 회의록 작성(네이버 - 클로바노트, 카카오 - Kakao i), 자막 생성(보이저엑스 - Vrew, 보이스루)

## Vectorize(벡터화)
- 벡터화: 컴퓨터는 자연어 자체를 받아들일 수 없기 때문에 컴퓨터가 이해할 수 있도록 벡터로 만들어주어야 한다.
- 자연어를 어떻게 벡터로 표현하는지는 자연어 처리 모델의 성능을 결정하는 중요한 역할이다.

### 벡터화의 2가지 방법

#### 등장 횟수 기반의 단어 표현(Count based Representation)
- 단어가 문서 혹은 문장에 등장하는 횟수를 기반으로 벡터화하는 방법
    - Bag of Words(CounterVectorizer)
    - TF-IDF(TfidfVectorizer)

#### 분포 기반의 단어 표현(Distributed Representation)
- 타겟 단어 주변에 있는 단어를 기반으로 벡터화하는 방법
    - Word2Vec
    - GloVe
    - fstText

# Text Preprocessing(텍스트 전처리)
- 자연어 처리의 시작이자 절반 이상을 차지하는 중요한 과정이다.
- 실제 텍스트 데이터를 다룰 때 데이터를 읽어보면서 어떤 특이사항이 있는지 파악해야 한다.
- 전처리 방법
    - 내장 메서드를 사용한 전처리(`lower`, `replace`)
    - 정규 표현식(Regular expression, Regex)
    - 불용어(Stop words) 처리
    - 통계적 트리밍(Trimming)
    - 어간 추출(Stemming), 표제어 추출(Lemmatization)

## 차원의 저주(Curse of Dimensionality)
- 차원의 저주: 특성의 개수가 선형적으로 늘어날 때 동일한 설명력을 가지기 위해 필요한 인스턴스의 수는 지수적으로 증가한다.
<br>즉, 동일한 개수의 인스턴스를 가지는 데이터셋의 차원이 늘어날수록 설명력이 떨어진다.
- 횟수 기반의 벡터 표현에서는 전체 말뭉치에 존재하는 단어의 종류가 데이터셋의 Feature, 즉 차원이 된다.
- 따라서, 단어의 종류를 줄여야 차원의 저주를 어느정도 해결할 수 있다.

## 대소문자 통일
- 모든 문자를 소문자로 통일하여 같은 범주로 엮는다.

```py
# sampling from: 'https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products'
import pandas as pd
df = pd.read_csv('https://ds-lecture-data.s3.ap-northeast-2.amazonaws.com/amazon/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_sample.csv')


# 데이터를 모두 소문자로 변환
df['brand'] = df['brand'].apply(lambda x: x.lower())
df['brand'].value_counts()
'''
amazon          5977
amazonbasics    4506
Name: brand, dtype: int64
'''
```

## Regex(정규표현식)
- 정규표현식: 구두점이나 특수문자 등 필요없는 문자가 말뭉치 내에 있을 경우 토큰화가 제대로 이루어지지 않기 때문에 이를 제거하기 위해 사용한다.
- 문자열을 다루기 위한 중요하고도 강력하지만, 복잡하기 때문에 반복적인 실습으로 익혀야 한다.
    - [Python Regex](https://www.w3schools.com/python/python_regex.asp#sub)
    - [정규 표현식 시작](https://wikidocs.net/4308#_2)
- `a-z`(소문자), `A-Z`(대문자), `0-9`(숫자)를 `^` 제외한 나머지 문자를 `regex` 에 할당한 후 `.sub` 메서드를 통해서 공백 문자열 `""` 로 치환한다.

```py
# 파이썬 정규표현식 패키지 이름: re
import re

# 정규식
# []: [] 사이 문자를 매치, ^: not
regex = r"[^a-zA-Z0-9 ]"

# 정규식을 적용할 스트링
test_str = ("(Natural Language Processing) is easy!, AI!\n")

# 치환할 문자
subst = ""

result = re.sub(regex, subst, test_str)
result # 구두점이나 특수문자가 사라짐
'''
Natural Language Processing is easy AI
'''

# 정규표현식을 수행한 후 소문자로 통일하고 공백 문자를 기준으로 나누는 함수 생성
def tokenize(text):
    """text 문자열을 의미있는 단어 단위로 list에 저장
    Args:
        text (str): 토큰화 할 문자열
    Returns:
        list: 토큰이 저장된 리스트
    """
    # 정규식 적용
    tokens = re.sub(regex, subst, text)

    # 소문자로 치환
    tokens = tokens.lower().split()
    
    return tokens


# 각 리뷰텍스트 토크나이즈 하여 tokens 칼럼 생성
df['tokens'] = df['reviews.text'].apply(tokenize)
df['tokens'].head()
'''
0    [though, i, have, got, it, for, cheap, price, ...
1    [i, purchased, the, 7, for, my, son, when, he,...
2    [great, price, and, great, batteries, i, will,...
3    [great, tablet, for, kids, my, boys, love, the...
4    [they, lasted, really, little, some, of, them,...
Name: tokens, dtype: object
'''


df[['reviews.text', 'tokens']].head(10)
'''

    reviews.text                                    	tokens
0	Though I have got it for cheap price during bl...	[though, i, have, got, it, for, cheap, price, ...
1	I purchased the 7" for my son when he was 1.5 ...	[i, purchased, the, 7, for, my, son, when, he,...
2	Great price and great batteries! I will keep o...	[great, price, and, great, batteries, i, will,...
3	Great tablet for kids my boys love their table...	[great, tablet, for, kids, my, boys, love, the...
4	They lasted really little.. (some of them) I u...	[they, lasted, really, little, some, of, them,...
5	I purchased 2 others for my 5 & 6yr-olds, and ...	[i, purchased, 2, others, for, my, 5, 6yrolds,...
6	We purchased Amazon Fire kids edition tablet t...	[we, purchased, amazon, fire, kids, edition, t...
7	Got this when they were on sale last year and ...	[got, this, when, they, were, on, sale, last, ...
8	Lotta batteries. at a good price.	                [lotta, batteries, at, a, good, price]
9	Best deal and work as expected	                    [best, deal, and, work, as, expected]
'''


# 결과 분석
from collections import Counter

# Counter 객체는 리스트요소의 값과 요소의 갯수를 카운트 하여 저장
# 카운터 객체는 .update 메소드로 계속 업데이트 가능
word_counts = Counter()

# 토큰화된 각 리뷰 리스트를 카운터 객체에 업데이트
df['tokens'].apply(lambda x: word_counts.update(x))

# 가장 많이 존재하는 단어 순으로 10개를 나열
word_counts.most_common(10)
'''
[('the', 10514),
 ('and', 8137),
 ('i', 7465),
 ('to', 7150),
 ('for', 6617),
 ('a', 6421),
 ('it', 6096),
 ('my', 4119),
 ('is', 4111),
 ('this', 3752)]
'''


# 위 코드 변형하여 코퍼스의 전체 워드 카운트, 랭크 등 정보가 담긴 DF를 리턴하는 함수를 구현하고 적용
def word_count(docs):
    """ 토큰화된 문서들을 입력받아 토큰을 카운트 하고 관련된 속성을 가진 데이터프레임을 리턴합니다.
    Args:
        docs (series or list): 토큰화된 문서가 들어있는 list
    Returns:
        list: Dataframe
    """
    # 전체 코퍼스에서 단어 빈도 카운트
    word_counts = Counter()

    # 단어가 존재하는 문서의 빈도 카운트, 단어가 한 번 이상 존재하면 +1
    word_in_docs = Counter()

    # 전체 문서의 갯수
    total_docs = len(docs)

    for doc in docs:
        word_counts.update(doc)
        word_in_docs.update(set(doc))

    temp = zip(word_counts.keys(), word_counts.values())

    wc = pd.DataFrame(temp, columns = ['word', 'count'])

    # 단어의 순위
    # method='first': 같은 값의 경우 먼저나온 요소를 우선
    wc['rank'] = wc['count'].rank(method='first', ascending=False)
    total = wc['count'].sum()

    # 코퍼스 내 단어의 비율
    wc['percent'] = wc['count'].apply(lambda x: x / total)

    wc = wc.sort_values(by='rank')

    # 누적 비율
    # cumsum() : cumulative sum
    wc['cul_percent'] = wc['percent'].cumsum()

    temp2 = zip(word_in_docs.keys(), word_in_docs.values())
    ac = pd.DataFrame(temp2, columns=['word', 'word_in_docs'])
    wc = ac.merge(wc, on='word')
    
    # 전체 문서 중 존재하는 비율
    wc['word_in_docs_percent'] = wc['word_in_docs'].apply(lambda x: x / total_docs)

    return wc.sort_values(by='rank')


wc = word_count(df['tokens'])
wc.head()
'''
    word	word_in_docs	count	rank	percent	    cul_percent	word_in_docs_percent
51	the	    4909	        10514	1.0	    0.039353	0.039353	0.468282
1	and	    5064	        8137	2.0	    0.030456	0.069809	0.483068
26	i	    3781	        7465	3.0	    0.027941	0.097750	0.360679
123	to	    4157        	7150	4.0	    0.026762	0.124512	0.396547
19	for	    4477	        6617	5.0	    0.024767	0.149278	0.427072
'''

wc[wc['rank'] <= 1000]['cul_percent'].max()
'''
0.9097585076280484
'''


# 단어 누적 분포 그래프
import seaborn as sns

sns.lineplot(x='rank', y='cul_percent', data=wc);
```

![image](https://user-images.githubusercontent.com/79494088/138678453-30f332a8-0b3b-48ad-8afd-d58c3265d0ba.png)

```py
# Squarify 라이브러리 사용 등장 비율 상위 20개 단어의 결과 시각화
!pip install squarify


import squarify
import matplotlib.pyplot as plt

wc_top20 = wc[wc['rank'] <= 20]
squarify.plot(sizes=wc_top20['percent'], label=wc_top20['word'], alpha=0.6)
plt.axis('off')
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/138678712-c30d1f9e-8682-404c-8f16-59ecd5a4a6d6.png)

## SqpCy
- `SpaCy`: 문서 구성요소를 다양한 구조에 나누어 저장하지 않고 요소를 색인화하여 검색 정보를 간단히 저장하는 라이브러리

```py
import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)

# 토큰화를 위한 파이프라인 구성
tokens = []

for doc in tokenizer.pipe(df['reviews.text']):
    doc_tokens = [re.sub(r"[^a-z0-9]", "", token.text.lower()) for token in doc]
    tokens.append(doc_tokens)

df['tokens'] = tokens
df['tokens'].head()
'''
0    [though, i, have, got, it, for, cheap, price, ...
1    [i, purchased, the, 7, for, my, son, when, he,...
2    [great, price, and, great, batteries, i, will,...
3    [great, tablet, for, kids, my, boys, love, the...
4    [they, lasted, really, little, some, of, them,...
Name: tokens, dtype: object
'''


# word_count 함수 사용 단어 분포 확인
wc = word_count(df['tokens'])
wc.head()
'''
	word	word_in_docs	count	rank	percent	    cul_percent	word_in_docs_percent
51	the	    4909	        10514	1.0	    0.039229	0.039229	0.468282
1	and	    5064	        8137	2.0	    0.030360	0.069589	0.483068
26	i	    3781	        7465	3.0	    0.027853	0.097442	0.360679
124	to	    4157	        7150	4.0	    0.026678	0.124120	0.396547
19	for	    4477	        6617	5.0	    0.024689	0.148809	0.427072
'''


# 등장 비율 상위 20개 단어 결과 시각화
wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['percent'], label=wc_top20['word'], alpha=0.6 )
plt.axis('off')
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/138679388-ce6d57c5-9020-4ea3-aba7-92269e458d21.png)

## 불용어(Stop words) 처리
- 위에서 시각화 한 I, and, of 같은 단어는 리뷰 관점에서 아무런 의미가 없는데, 이런 단어들을 불용어라고 한다.
- 대부분의 NLP 라이브러리는 접속사, 관사, 부사, 대명사, 일반동사 등을 포함한 일반적인 불용어를 내장하고 있다.

```py
# spacy가 기본적으로 제공하는 불용어
print(nlp.Defaults.stop_words)
{'could', 'to', 'others', 'herein', 'cannot', 'whereafter', 'through', 'hers', 'while', 'being', 'a', '‘m', 'together', 'noone', "'ve", 'became', 'seems', 'somewhere', 'why', 'did', 'an', 'have', 'something', 'regarding', 'among', 'as', 'show', 'from', "n't", 'of', 'which', 'less', 'thus', 'himself', 'after', 'whose', 'whole', '‘ll', 'full', 'once', 'only', 'ever', 'becomes', 'whereupon', 'nevertheless', 'every', 'up', 'him', 'myself', 'alone', 'somehow', 'few', 'on', 'must', 'done', 'the', 'serious', 'fifteen', 'now', 'hereby', 'therefore', 'my', 'ourselves', 'been', 'some', 'down', 'forty', 'take', 'and', 'again', 'that', 'about', 'though', 'everyone', 'not', 'via', 'really', 'nothing', 'more', 'mostly', 'made', 'before', 'so', 'four', 'fifty', 'never', 'nor', 'if', 'further', 'most', 'several', 'anyway', 'beyond', 'unless', '‘ve', 'all', 'therein', 'can', 'during', 'hundred', 'anything', "'re", 'beforehand', 'in', 'toward', 'amount', 'however', 'least', 'where', 'they', 'nobody', 'very', 'empty', 'below', 'might', 'with', 'twelve', 'will', 'may', 'except', 'give', 'latterly', 'is', 'along', 'either', 'become', 'ca', 'off', 'because', 'perhaps', 'your', 'too', 'do', 'keep', 'us', 'these', 'call', 'say', 'our', 'mine', 'ten', 'both', 'those', '’re', 'other', 'indeed', 'twenty', 'quite', 'latter', 'rather', 'am', 'thereby', 'against', 'at', 'sometime', 'does', 'side', 'n‘t', 'bottom', 'same', 'next', 'hence', 'under', 'such', 'due', 'whom', 'her', 'doing', "'d", 'themselves', 'much', 'thence', 'put', 'beside', 'name', 'who', 'be', "'m", 'top', 'would', 'he', 'formerly', 'sixty', 'above', 'first', 'used', 'what', 'whoever', 'former', 'are', 'get', 'but', 'without', '’d', 'or', 'meanwhile', 'just', 'between', 're', 'whereas', 'then', 'someone', 'ours', 'well', 'was', 'throughout', 'sometimes', 'across', 'thru', 'please', 'enough', 'upon', 'otherwise', "'s", 'back', 'seemed', 'often', 'already', 'everywhere', 'when', '’m', 'go', 'whence', 'onto', 'many', "'ll", 'seem', 'this', 'yours', 'herself', 'whither', 'none', 'should', 'i', 'no', 'yourself', 'although', 'behind', 'until', 'me', 'whatever', 'part', '’ll', 'amongst', 'thereafter', 'yet', 'becoming', 'out', 'nowhere', 'front', 'five', 'seeming', 'own', 'everything', 'else', 'it', 'wherever', 'afterwards', 'two', 'itself', 'nine', 'its', 'how', 'almost', 'make', '‘s', 'hereupon', 'one', 'his', 'eleven', 'thereupon', 'various', 'anyone', 'any', 'by', 'than', 'also', 'using', 'here', 'were', 'around', 'see', 'moreover', 'third', 'whether', 'neither', 'we', 'eight', 'into', 'you', 'for', '’ve', 'namely', 'always', 'last', 'their', '‘d', 'hereafter', 'towards', 'had', 'even', 'has', 'she', 'six', 'since', 'elsewhere', 'each', 'anyhow', '’s', 'whenever', 'there', 'anywhere', '‘re', 'still', 'yourselves', 'move', 'wherein', 'whereby', 'another', 'three', 'per', 'over', 'besides', 'within', 'n’t', 'them'}


# 해당 불용어 제외하고 토크나이징을 진행한 결과
tokens = []
# 토큰에서 불용어 제거, 소문자화 하여 업데이트
for doc in tokenizer.pipe(df['reviews.text']):
    doc_tokens = []

    # A doc is a sequence of Token(<class 'spacy.tokens.doc.Doc'>)
    for token in doc:
        # 토큰이 불용어와 구두점이 아니면 저장
        if (token.is_stop == False) & (token.is_punct == False):
            doc_tokens.append(token.text.lower())

    tokens.append(doc_tokens)

df['tokens'] = tokens
df.tokens.head()
'''
0    [got, cheap, price, black, friday,, fire, grea...
1    [purchased, 7", son, 1.5, years, old,, broke, ...
2    [great, price, great, batteries!, buying, anyt...
3         [great, tablet, kids, boys, love, tablets!!]
4    [lasted, little.., (some, them), use, batterie...
Name: tokens, dtype: object
'''


wc = word_count(df['tokens'])

wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['percent'], label=wc_top20['word'], alpha=0.6)
plt.axis('off')
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/138680167-db5dcc52-fcd2-4ea1-8555-52bb3977f0d0.png)

### 불용어 커스터마이징

```py
STOP_WORDS = nlp.Defaults.stop_words.union(['batteries','I', 'amazon', 'i', 'Amazon', 'it', "it's", 'it.', 'the', 'this'])

tokens = []

for doc in tokenizer.pipe(df['reviews.text']):
    
    doc_tokens = []
    
    for token in doc: 
        if token.text.lower() not in STOP_WORDS:
            doc_tokens.append(token.text.lower())
   
    tokens.append(doc_tokens)
    
df['tokens'] = tokens


wc = word_count(df['tokens'])
wc.head()
'''

    word	word_in_docs	count	rank	percent	    cul_percent	word_in_docs_percent
58	great	2709	        3080	1.0	    0.024609	0.024609	0.258418
14	good	1688        	1870	2.0	    0.014941	0.039549	0.161023
68	tablet	1469	        1752	3.0	    0.013998	0.053547	0.140132
64	love	1183	        1287	4.0	    0.010283	0.063830	0.112849
103	bought	1103	        1179	5.0	    0.009420	0.073250	0.105218
'''


wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['percent'], label=wc_top20['word'], alpha=0.6)
plt.axis('off')
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/138680440-831f6bd9-1e83-4048-b4e7-26b19433ca9a.png)

```py
# 킨들의 리뷰와 전체 리뷰의 토큰 비교
df['kindle'] = df['name'].str.contains('kindle', case=False)
wc_kindle = word_count(df[df['kindle'] == 1]['tokens'])


wc_top20 = wc[wc['rank'] <= 20]
wc_kindle_top20 = wc_kindle[wc_kindle['rank'] <= 20]

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
axes[0].set_title('All Reviews')
squarify.plot(sizes=wc_top20['percent'], label=wc_top20['word'], alpha=0.6, ax=axes[0], text_kwargs={'fontsize':14})
axes[0].axis('off')

axes[1].set_title('Kindle Reviews')
squarify.plot(sizes=wc_kindle_top20['percent'], label=wc_kindle_top20['word'], alpha=0.6, ax=axes[1], text_kwargs={'fontsize':14})
axes[1].axis('off')

plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/138680542-dcba6ad3-5aae-4042-a6b1-36eeddabf7b9.png)

## 통계적 Trimming
- 불용어를 직접적으로 제거하는 대신 통계적인 방법을 통해 말뭉치 내에서 많거나, 적은 토큰을 제거하는 방법도 있다.

```py
# 단어 누적 분포 그래프
sns.lineplot(x='rank', y='cul_percent', data=wc)
```

![image](https://user-images.githubusercontent.com/79494088/138680693-ec5df5ad-287e-4003-946d-49fc7125ec4d.png)

- 몇몇 소수의 단어들이 전체 코퍼스의 80%를 차지한다는 점을 알 수 있다.
- 그래프 결과에서 나타는 단어의 중요도의 두 가지 해석
    - 자주 나타나는 단어들(그래프의 왼쪽): 여러 문서에서 두루 나타나기 때문에 문세 분류 단계에서 통찰력을 제공하지 않는다.
    - 자주 나타나지 않는 단어들(그래프의 오른쪽): 너무 드물게 나타나기 때문에 큰 의미가 없을 확률이 높다.

```py
# 랭크가 높거나 낮은 단어
wc.tail(20)
```

![image](https://user-images.githubusercontent.com/79494088/138681823-08c85a3e-a11b-4a82-98c5-280355b06473.png)

```py
wc['word_in_docs_percent'].describe()
'''
count    13497.000000
mean         0.000838
std          0.004632
min          0.000095
25%          0.000095
50%          0.000095
75%          0.000382
max          0.258418
Name: word_in_docs_percent, dtype: float64
'''


wc['word_in_docs_percent']
'''
58       0.258418
14       0.161023
68       0.140132
64       0.112849
103      0.105218
           ...   
13492    0.000095
13493    0.000095
13494    0.000095
13495    0.000095
13496    0.000095
Name: word_in_docs_percent, Length: 13497, dtype: float64
'''


# 문서에 나타나는 빈도
sns.displot(wc['word_in_docs_percent'],kind='kde')
```

![image](https://user-images.githubusercontent.com/79494088/138681931-b6adb811-13c3-43e6-8078-80cea056a9f5.png)

```py
# 최소한 1% 이상 문서에 나타나는 단어만 선택
wc = wc[wc['word_in_docs_percent'] >= 0.01]
sns.displot(wc['word_in_docs_percent'], kind='kde');
```

![image](https://user-images.githubusercontent.com/79494088/138682072-c922bd47-9bbe-4a1c-92c7-0e6a91dd9099.png)

```py
wc
```

![image](https://user-images.githubusercontent.com/79494088/138682181-279d2096-8341-422d-8f3d-9cac8d771429.png)

## 어간 추출(Stemming)과 표제어 추출(Lemmatization)

### 어간 추출(Stemming)
- 어간(Stem): 단어의 의미가 포함된 부분으로 접사등이 제거된 형태
    - 어근이나 단어의 원형과 같지 않을 수 있다.
    - ex) argue, argued, arguing, argus의 어간은 단어의 뒷 부분이 제거된 argu가 어간이다.
- 어간 추출은 ing, ed, s 등과 같은 부분을 제거하게 된다.

#### nltk 사용 Stemming

```py
from nltk.stem import PorterStemmer

ps = PorterStemmer()

words = ["wolf", "wolves"]

for word in words:
    print(ps.stem(word))
'''
wolf
wolv
'''


# 아마존 리뷰 데이터
tokens = []
for doc in df['tokens']:
    doc_tokens = []
    for token in doc:
        doc_tokens.append(ps.stem(token))
    tokens.append(doc_tokens)

df['stems'] = tokens


wc = word_count(df['stems'])

wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['percent'], label=wc_top20['word'], alpha=0.6 )
plt.axis('off')
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/138694026-bb3d42cd-bbd0-4d84-8543-0a2a209b9035.png)

- 단지 단어의 끝 부분을 자르는 역할을 하기 때문에 사전에도 없는 단어가 많이 나오는데, 이상하긴 해도 현실적으로 사용하기에 성능이 나쁘지 않다.
- 알고리즘이 간단하여 속도가 빠르기 때문에 속도가 중요한 검색 분야에서 많이 사용한다.

### 표제어 추출(Lemmatization)
- 어간 추출보다 체계적이다.
- 단어의 기본 사전형 단어 형태인 Lemma(표제어)로 변환된다.
- 명사의 복수형은 단수형으로, 동사는 모두 타동사로 변환된다.
- 단어로부터 표제어를 찾아가는 과정은 Stemming 보다 많은 연산이 필요하다.

#### SpaCy 사용 Lemmatization

```py
lem = "The social wolf. Wolves are complex."

nlp = spacy.load("en_core_web_sm")

doc = nlp(lem)

# wolf, wolve가 어떤 Lemma로 추출되는지 확인해 보세요
for token in doc:
    print(token.text, "  ", token.lemma_)
'''
The    the
social    social
wolf    wolf
.    .
Wolves    wolf
are    be
complex    complex
.    .
'''


# 과정 함수화
def get_lemmas(text):
    lemmas = []
    doc = nlp(text)
    for token in doc: 
        if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ != 'PRON'):
            lemmas.append(token.lemma_)
    
    return lemmas


df['lemmas'] = df['reviews.text'].apply(get_lemmas)
df['lemmas'].head()
'''
0    [get, cheap, price, black, friday, fire, great...
1    [purchase, 7, son, 1.5, year, old, break, wait...
2    [great, price, great, battery, buy, anytime, n...
3              [great, tablet, kid, boy, love, tablet]
4    [last, little, use, battery, lead, lamp, 2, 4,...
Name: lemmas, dtype: object
'''


wc = word_count(df['lemmas'])
wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['percent'], label=wc_top20['word'], alpha=0.6 )
plt.axis('off')
plt.show()
```

![image](https://user-images.githubusercontent.com/79494088/138694676-47c01222-8dc5-40f2-a189-6d65258b4c5d.png)

# 등장 횟수 기반의 단어 표현(Count based Representation)

## 텍스트 문서 벡터화
- 머신러닝 모델에서 텍스트를 분석하기 위해서는 벡터화(Vectorization)하는 과정이 필요하다.
- 벡터화: 텍스트를 컴퓨터가 계산할 수 있도록 수치정보로 변환하는 과정이다.
- 등상 횟수 기반의 단어 표현(Count based Representation)은 단어가 특정 문서혹은 문장에 들어있는 횟수를 바탕으로 해당 문서를 벡터화한다.
- 대표적인 방법으로 Bag of Words(TF, TF-IDF) 방식이 있다.

### 문서-단어 행렬(Document Term Matrix, DTM)
- 벡터화 된 문서는 문서-단어 행렬의 형태로 나타내어진다.
- 문서-단어 행렬이란 각 행에는 문서(Document)가, 각 열에는 단어(Term)가 있는 행렬이다.

|        | Word_1 | Word_2 | Word_3 | Word_4 | Word_5 | Word_6 |
| **Docu_1** | 1      | 2      | 0      | 1      | 0      | 0      |
| **Docu_2** | 0      | 0      | 0      | 1      | 1      | 1      |
| **Docu_3** | 1      | 0      | 0      | 1      | 0      | 1      |

## Bag of Words(BoW): TF(Term Frequency)
- BoW는 가장 단순환 벡터화 방법 중 하나이다.
- 문서 혹은 문장에서 문법이나 단어의 순서 등을 무시하고 단순히 단어의 빈도만 고려하여 벡터화한다.

![image](https://user-images.githubusercontent.com/79494088/138695602-95926cf0-5e82-45f4-a68d-0b92a2ac152b.png)

### CountVectorizer 적용

```py
# 모듈에서 사용할 라이브러리와 spacy 모델 import
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import spacy
nlp = spacy.load("en_core_web_sm")


# 예제로 사용할 text 선언
text = """In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.
It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling.
The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word,
which helps to adjust for the fact that some words appear more frequently in general.
tf–idf is one of the most popular term-weighting schemes today.
A survey conducted in 2015 showed that 83% of text-based recommender systems in digital libraries use tf–idf."""


# spacy 언어모델 이용 token화된 단어 확인
doc = nlp(text)
print([token.lemma_ for token in doc if (token.is_stop != True) and (token.is_punct != True)])
'''
['information', 'retrieval', 'tf', 'idf', 'TFIDF', 'short', 'term', 'frequency', 'inverse', 'document', 'frequency', 'numerical', 'statistic', 'intend', 'reflect', 'important', 'word', 'document', 'collection', 'corpus', '\n', 'weight', 'factor', 'search', 'information', 'retrieval', 'text', 'mining', 'user', 'modeling', '\n', 'tf', 'idf', 'value', 'increase', 'proportionally', 'number', 'time', 'word', 'appear', 'document', 'offset', 'number', 'document', 'corpus', 'contain', 'word', '\n', 'help', 'adjust', 'fact', 'word', 'appear', 'frequently', 'general', '\n', 'tf', 'idf', 'popular', 'term', 'weight', 'scheme', 'today', '\n', 'survey', 'conduct', '2015', 'show', '83', 'text', 'base', 'recommender', 'system', 'digital', 'library', 'use', 'tf', 'idf']
'''


from sklearn.feature_extraction.text import CountVectorizer

# 문장으로 이루어진 리스트 저장
sentences_lst = text.split('\n')

# CountVectorizer 변수 저장
vect = CountVectorizer()

# 어휘 사전 생성합니다.
vect.fit(sentences_lst)
'''
CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
                lowercase=True, max_df=1.0, max_features=None, min_df=1,
                ngram_range=(1, 1), preprocessor=None, stop_words=None,
                strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                tokenizer=None, vocabulary=None)
'''


# text를 DTM(document-term matrix) 변환(transform)
dtm_count = vect.transform(sentences_lst)


# .vocabulary_메서드 활용 모든 토큰과 맵핑된 인덱스 정보 확인
'''
{'2015': 0,
 '83': 1,
 'adjust': 2,
 'and': 3,
 'appear': 4,
 'appears': 5,
 'as': 6,
 'based': 7,
 'by': 8,
 'collection': 9,
 'conducted': 10,
 'contain': 11,
 'corpus': 12,
 'digital': 13,
 'document': 14,
 'documents': 15,
 'fact': 16,
 'factor': 17,
 'for': 18,
 'frequency': 19,
 'frequently': 20,
 'general': 21,
 'helps': 22,
 'how': 23,
 'idf': 24,
 'important': 25,
 'in': 26,
 'increases': 27,
 'information': 28,
 'intended': 29,
 'inverse': 30,
 'is': 31,
 'it': 32,
 'libraries': 33,
 'mining': 34,
 'modeling': 35,
 'more': 36,
 'most': 37,
 'number': 38,
 'numerical': 39,
 'of': 40,
 'offset': 41,
 'often': 42,
 'one': 43,
 'or': 44,
 'popular': 45,
 'proportionally': 46,
 'recommender': 47,
 'reflect': 48,
 'retrieval': 49,
 'schemes': 50,
 'searches': 51,
 'short': 52,
 'showed': 53,
 'some': 54,
 'statistic': 55,
 'survey': 56,
 'systems': 57,
 'term': 58,
 'text': 59,
 'tf': 60,
 'tfidf': 61,
 'that': 62,
 'the': 63,
 'times': 64,
 'to': 65,
 'today': 66,
 'use': 67,
 'used': 68,
 'user': 69,
 'value': 70,
 'weighting': 71,
 'which': 72,
 'word': 73,
 'words': 74}
'''


dtm_count.shape
'''
(6, 75)
'''


# 추출된 토큰 나열
print(f"""
features : {vect.get_feature_names()}
# of features : {len(vect.get_feature_names())}
""")
'''
features : ['2015', '83', 'adjust', 'and', 'appear', 'appears', 'as', 'based', 'by', 'collection', 'conducted', 'contain', 'corpus', 'digital', 'document', 'documents', 'fact', 'factor', 'for', 'frequency', 'frequently', 'general', 'helps', 'how', 'idf', 'important', 'in', 'increases', 'information', 'intended', 'inverse', 'is', 'it', 'libraries', 'mining', 'modeling', 'more', 'most', 'number', 'numerical', 'of', 'offset', 'often', 'one', 'or', 'popular', 'proportionally', 'recommender', 'reflect', 'retrieval', 'schemes', 'searches', 'short', 'showed', 'some', 'statistic', 'survey', 'systems', 'term', 'text', 'tf', 'tfidf', 'that', 'the', 'times', 'to', 'today', 'use', 'used', 'user', 'value', 'weighting', 'which', 'word', 'words']
# of features : 75
'''


# CountVectorizer로 제작한 dtm 분석
print(type(dtm_count))
print(dtm_count)
'''
<class 'scipy.sparse.csr.csr_matrix'>
  (0, 9)	1
  (0, 12)	1
  (0, 14)	2
  (0, 18)	1
  (0, 19)	2
  (0, 23)	1
  (0, 24)	1
  (0, 25)	1
  (0, 26)	2
  (0, 28)	1
  (0, 29)	1
  (0, 30)	1
  (0, 31)	3
  (0, 39)	1
  (0, 44)	2
  (0, 48)	1
  (0, 49)	1
  (0, 52)	1
  (0, 55)	1
  (0, 58)	1
  (0, 60)	1
  (0, 61)	1
  (0, 62)	1
  (0, 65)	2
  (0, 73)	1
  :	:
  (4, 43)	1
  (4, 45)	1
  (4, 50)	1
  (4, 58)	1
  (4, 60)	1
  (4, 63)	1
  (4, 66)	1
  (4, 71)	1
  (5, 0)	1
  (5, 1)	1
  (5, 7)	1
  (5, 10)	1
  (5, 13)	1
  (5, 24)	1
  (5, 26)	2
  (5, 33)	1
  (5, 40)	1
  (5, 47)	1
  (5, 53)	1
  (5, 56)	1
  (5, 57)	1
  (5, 59)	1
  (5, 60)	1
  (5, 62)	1
  (5, 67)	1
'''


# dtm_count의 타입: CSR(Compressed Sparse Row matrix)
# 해당 타입은 행렬(matrix)에서 0을 표현하지 않는 타입

# .todense() 메서드 사용 numpy.matrix 타입 반환

print(type(dtm_count))
print(type(dtm_count.todense()))
dtm_count.todense()
'''
<class 'scipy.sparse.csr.csr_matrix'>
<class 'numpy.matrix'>
matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 2, 0,
         0, 0, 1, 1, 1, 2, 0, 1, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 2, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1,
         0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0,
         1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 2, 1,
         0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
         6, 1, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
         1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
         1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
         0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
         1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1,
         0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
'''


# DataFrame 변환 후 결과값 확인
dtm_count = pd.DataFrame(dtm_count.todense(), columns=vect.get_feature_names())
print(type(dtm_count))

dtm_count
```

![image](https://user-images.githubusercontent.com/79494088/138699195-611f24f4-c2bc-4859-97ed-63c5335dbf7a.png)

```py
# 아마존 리뷰 데이터에 CountVectorizer 적용
from sklearn.feature_extraction.text import CountVectorizer


count_vect = CountVectorizer(stop_words='english', max_features=100)

# Fit 후 dtm 생성(문서, 단어마다 tf-idf 값 계산)
dtm_count_amazon = count_vect.fit_transform(df['reviews.text'])

dtm_count_amazon = pd.DataFrame(dtm_count_amazon.todense(), columns=count_vect.get_feature_names())
dtm_count_amazon
```

![image](https://user-images.githubusercontent.com/79494088/138699651-9f6651b4-da7a-4112-98c0-49d72ea403ec.png)

## Bag of Words(BoW): TF-IDF(Term Frequency - Inverse Document Frequency)
- 각 문서마다 중요한 단어와 그렇지 못한 단어가 있다.
- 다른 문서에 잘 등장하지 않는 단어라면 해당 문서를 대표할 수 있는 단어가 될 수 있다.
- 다른 문서에 등장하지 않는 단어, 즉 특정 문서에만 등장하는 단어에 가중치를 두는 방법: TF-IDF(Term Frequency-Inverse Document Frequency)

### 수식

$$
\text{TF-IDF(w)} = \text{TF(w)} \times \text{IDF(w)}
$$

- TF(Term-Frequency)는 특정 문서에만 단어 w가 쓰인 빈도이다.
- 분석할 문서에서 단어 $w$가 등장하는 횟수를 구하게 된다.

$$
\text{TF(w)} = \text{특정 문서 내 단어 w의 수}
$$

- IDF(Inverse Document Frequency)는 분류 대상이 되는 모든 문서의 수를 단어 $w$가 들어있는 문서의 수로 나누어 준 뒤 로그를 취해준 값이다.

$$
\text{IDF(w)} = \log \bigg(\frac{\text{분류 대상이 되는 모든 문서의 수}}{\text{단어 w가 들어있는 문서의 수}}\bigg)
$$

- 실제 계산에서는 0으로 나누어 주는 것을 방지하기 위해 분모에 1을 더해준 값을 사용한다.

$$
\text{분류 대상이 되는 모든 문서의 수} : n \\
\text{단어 w가 들어있는 문서의 수} : df(w)
$$

- 즉,

$$
\text{IDF(w)} = \log \bigg(\frac{n}{1 + df(w)}\bigg)
$$

- 위 식에 따르면 자주 사용하는 단어라도, 많은 문서에 나오는 단어는 IDF가 낮아지기 때문에 TF-IDF가 낮아지기 때문에 TF-IDF로 벡터화 했을 때 작은 값을 가지게 된다.(지프의 법칙 찾아보기. log 이유 찾아보기)
- 사이킷런의 TfidfVectorizer를 사용하면 TF-IDF 벡터화도 사용할 수 있다.

### TfidVectorizer 적용

```py
# TF-IDF vectorizer. 테이블을 작게 만들기 위해 max_features=15로 제한
tfidf = TfidfVectorizer(stop_words='english', max_features=15)

# Fit 후 dtm 생성(문서, 단어마다 tf-idf 값을 계산)
dtm_tfidf = tfidf.fit_transform(sentences_lst)

dtm_tfidf = pd.DataFrame(dtm_tfidf.todense(), columns=tfidf.get_feature_names())


# TfidVectorizer를 사용하여 생성한 문서-단어 행렬(DTM)의 값을 CountVectorizer를 사용하여 생성한 DTM의 값과 비교
vect = CountVectorizer(stop_words='english', max_features=15)
dtm_count_vs_tfidf = vect.fit_transform(sentences_lst)
dtm_count_vs_tfidf = pd.DataFrame(dtm_count_vs_tfidf.todense(), columns=vect.get_feature_names())
dtm_count_vs_tfidf
```

![image](https://user-images.githubusercontent.com/79494088/138705778-1f78bca3-6182-4518-b83c-55729fec906f.png)

```py
# 파라미터 튜닝: SpaCy tokenizer 사용 벡터화 진행
def tokenize(document):
    doc = nlp(document)
    return [token.lemma_.strip() for token in doc if (token.is_stop != True) and (token.is_punct != True) and (token.is_alpha == True)]


"""
args:
    ngram_range = (min_n, max_n), min_n 개~ max_n 개를 갖는 n-gram(n개의 연속적인 토큰)을 토큰으로 사용합니다.
    min_df = n : int, 최소 n개의 문서에 나타나는 토큰만 사용합니다.
    max_df = m : float(0~1), m * 100% 이상 문서에 나타나는 토큰은 제거합니다.
"""

tfidf_tuned = TfidfVectorizer(stop_words='english'
                        ,tokenizer=tokenize
                        ,ngram_range=(1,2)
                        ,max_df=.7
                        ,min_df=3
                       )

dtm_tfidf_tuned = tfidf_tuned.fit_transform(df['reviews.text'])
dtm_tfidf_tuned = pd.DataFrame(dtm_tfidf_tuned.todense(), columns=tfidf_tuned.get_feature_names())
dtm_tfidf_tuned.head()
```

![image](https://user-images.githubusercontent.com/79494088/138706420-99308663-cff8-4d12-9d95-2928c43d3758.png)

(n-gram 찾아보기)

```py
# 아마존 리뷰 데이터에 tfidfVectorizer 적용
tfidf_vect = TfidfVectorizer(stop_words='english', max_features=100)

# Fit 후 dtm 생성(문서, 단어마다 tf-idf 값 계산)
dtm_tfidf_amazon = tfidf_vect.fit_transform(df['reviews.text'])

dtm_tfidf_amazon = pd.DataFrame(dtm_tfidf_amazon.todense(), columns=tfidf_vect.get_feature_names())
dtm_tfidf_amazon
```

![image](https://user-images.githubusercontent.com/79494088/138707373-7141c96a-97dd-4201-988e-36cb5d502b80.png)

(CountVectorizer, TfidfVectorizer 차이와 장단점 찾아보기)

### 유사도 이용 문서 검색
- 검색엔진의 원리: 검색어와 문서에 있는 단어를 매칭하여 결과를 보여준다.
- 매칭의 방법 중 가장 클래식한 방법인 유사인 측정 방법을 시도해보자.

$$ \text{cosine similarity} = cos (\theta)=\frac{A⋅B}{\Vert A\Vert \Vert B \Vert} $$

![image](https://user-images.githubusercontent.com/79494088/138708410-a2182c96-886f-4009-a829-62ff28231657.png)

- 코사인 유사도는 두 벡터가 이루는 각의 코사인 값을 이용하여 구할 수 있는 유사도이다.
- 두 벡터가
    - 완전히 같을 경우 1
    - 90도의 각을 이루면 0
    - 완전히 반대방향을 이루면 -1

#### NearestNeighbor(K-NN, K-최근접 이웃)
- K-최근접 이웃법은 쿼리와 가장 가가운 상위 K개의 근접한 데이터를 찾아서 K개 데이터의 유사성을 기반으로 점을 추정하거나 분류하는 예측 분석에 사용된다.

```py
from sklearn.neighbors import NearestNeighbors

# dtm 사용 NN 모델 학습 (디폴트)최근접 5 이웃
nn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')
nn.fit(dtm_tfidf_amazon)
'''
NearestNeighbors(algorithm='kd_tree', leaf_size=30, metric='minkowski',
                 metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                 radius=1.0)
'''


# 2번째 인덱스에 해당하는 문서와 가장 가까운 문서(0 포함) 5개의 거리(값이 작을수록 유사)와 문서의 인덱스를 알 수 있다.
nn.kneighbors([dtm_tfidf_amazon.iloc[2]])
'''
(array([[0.        , 0.64660432, 0.73047367, 0.76161463, 0.76161463]]),
 array([[   2, 7278, 6021, 1528, 4947]]))
'''


# 2번째 인덱스 문서의 이웃인 7278번째 인덱스 문서로 검색
nn.kneighbors([dtm_tfidf_amazon.iloc[7278]])
'''
(array([[0.        , 0.43712229, 0.58937603, 0.58937603, 0.58937603]]),
 array([[7278, 6021, 7714, 3216,  746]]))
'''


print(df['reviews.text'][2][:300])
print(df['reviews.text'][7278][:300])
'''
Great price and great batteries! I will keep on buying these anytime I need more!
Always need batteries and these come at a great price.
'''


# 문서 검색 예제: Amazon Review의 Sample을 가져와서 문서검색에 사용
# 출처 : https://www.amazon.com/Samples/product-reviews/B000001HZ8?reviewerType=all_reviews
sample_review = ["""in 1989, I managed a crummy bicycle shop, "Full Cycle" in Boulder, Colorado.
The Samples had just recorded this album and they played most nights, at "Tulagi's" - a bar on 13th street.
They told me they had been so broke and hungry, that they lived on the free samples at the local supermarkets - thus, the name.
i used to fix their bikes for free, and even feed them, but they won't remember.
That Sean Kelly is a gifted songwriter and singer."""]


# 학습된 TfidfVectorizer를 통해 Sample Review를 변환
new = tfidf_vect.transform(sample_review)


nn.kneighbors(new.todense())
'''
(array([[0.69016304, 0.81838594, 0.83745037, 0.85257729, 0.85257729]]),
 array([[10035,  2770,  1882,  9373,  3468]]))
'''


# 가장 가깝게 나온 문서 확인
df['reviews.text'][10035]
'''
Doesn't get easier than this. Good products shipped to my office free, in two days:)
'''
```

# Review

## 불용어(Stopwords)
- 자주 등장하지만 분석을 함에 있어서 크게 도움되지 않는 단어이다.
- NLTK에서 100여개 이상의 영어 단어를 불용어로 패키지 내에서 미리 정의하고 있다.

## 어간 추출(Stemming)과 표제어 추출(Lemmatization)

### 어간 추출
- 단어의 의미가 포함된 부분으로 접사 등이 제거된 형태로써, 사전에 존재하지 않는 단어일 수 있다.

### 표제어 추출
- 단어의 기본 사전형 단어 형태인 Lemma(표제어)로 변환된다.
- Stemming보다 많은 연산이 필요하다.

## TF-IDF

$$ idf(d, t) = log(\frac{n}{1+df(t)}) $$

- 단어의 빈도와 역 문서 빈도를 사용하여 DTM 내의 각 단어마다 중요한 정도를 가중치로 주는 방법이다.
- 유사도를 구하는 작업, 검색 결과의 중요도를 정하는 작업, 특정 단어의 중요도를 구하는 작업 등에 쓰인다.

## n-gram
- 모든 단어를 고려하는 것이 아닌 일부 단어만 고려하는 접근 방법인데, 이 때 일부 단어 몇 개보느냐를 결정하는 것이 n이 가지는 의미이다.
- 단어 몇 개만 보다 보니 의도하고 싶은 대로 문장을 끝맺음 하지 못하는 경우가 생긴다.

# References
- [Spacy 101](https://course.spacy.io)
- [regex101.com(정규식 연습)](https://regex101.com/)
- [Python RegEx](https://www.w3schools.com/python/python_regex.asp#sub)
- [정규 표현식 시작하기](https://wikidocs.net/4308#_2)