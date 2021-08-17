---
title: '[ML] 결정트리(Decision Trees) 이용 H1N1 모델링 캐글 첫 제출'
description: H1N1 데이터를 결정트리를 이용해서 모델링하고 캐글에 제출
categories:
 - Machine Learning
tags: [Did Unknown, Machine Learning, Decision Trees, Scikit-learn, Pipelines, 사이킷런, 파이프라인, 결정트리]
mathjax: enable
# 0️⃣1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣8️⃣9️⃣🔟
---

# 1️⃣ EDA

```py
!pip install category_encoders # 카테고리 인코더스 설치
!pip install pandas-profiling==2.11.0 # 프로파일링 설치

import pandas as pd
from sklearn.model_selection import train_test_split

target = 'vacc_h1n1_f' # 타겟 설정

train = pd.merge(pd.read_csv('train.csv'), 
                 pd.read_csv('train_labels.csv')[target], left_index=True, right_index=True)
test = pd.read_csv('test.csv')

train.head().T
```

![스크린샷 2021-08-17 20 12 59](https://user-images.githubusercontent.com/79494088/129716390-4078f698-b284-4542-906a-9fec0fc5c348.png)

```py
train.dtypes

'''
h1n1_concern                   float64
h1n1_knowledge                 float64
behavioral_antiviral_meds      float64
behavioral_avoidance           float64
behavioral_face_mask           float64
behavioral_wash_hands          float64
behavioral_large_gatherings    float64
behavioral_outside_home        float64
behavioral_touch_face          float64
doctor_recc_h1n1               float64
doctor_recc_seasonal           float64
chronic_med_condition          float64
child_under_6_months           float64
health_insurance               float64
health_worker                  float64
opinion_h1n1_vacc_effective     object
opinion_h1n1_risk               object
opinion_h1n1_sick_from_vacc     object
opinion_seas_vacc_effective     object
opinion_seas_risk               object
opinion_seas_sick_from_vacc     object
agegrp                          object
education_comp                 float64
raceeth4_i                       int64
sex_i                            int64
inc_pov                          int64
marital                        float64
rent_own_r                     float64
employment_status               object
census_region                    int64
census_msa                      object
n_adult_r                      float64
household_children             float64
n_people_r                     float64
employment_industry             object
employment_occupation           object
hhs_region                       int64
state                           object
vacc_h1n1_f                      int64
dtype: object
'''

# 8:2 비율로 나눠줌
train, val = train_test_split(train, train_size=0.80, test_size=0.20, stratify=train[target], random_state=2) # stratify : 데이터가 계층화된 방식으로 분할되어 클래스 레이블로 사용

train.shape, val.shape, test.shape
'''
((33723, 39), (8431, 39), (28104, 38))
'''

train[target].value_counts(normalize=True) # 기준모델 : 0.76
'''
0    0.760935
1    0.239065
Name: vacc_h1n1_f, dtype: float64
'''

# Profiling으로 데이터 리포트 생성
from pandas_profiling import ProfileReport

profile = ProfileReport(train, minimal=True).to_notebook_iframe()
```

![스크린샷 2021-08-17 20 20 06](https://user-images.githubusercontent.com/79494088/129717303-542348ac-7c09-464f-8d14-ab756513f507.png)

```py
# 실수형 타입 확인
train.select_dtypes('float').head(20).T
```

![스크린샷 2021-08-17 20 21 05](https://user-images.githubusercontent.com/79494088/129717432-414d98ff-ba2e-40ce-9725-43cea260a253.png)

```py
# 중복된 특성 확인
train.T.duplicated()
'''
h1n1_concern                   False
h1n1_knowledge                 False
behavioral_antiviral_meds      False
behavioral_avoidance           False
behavioral_face_mask           False
behavioral_wash_hands          False
behavioral_large_gatherings    False
behavioral_outside_home        False
behavioral_touch_face          False
doctor_recc_h1n1               False
doctor_recc_seasonal           False
chronic_med_condition          False
child_under_6_months           False
health_insurance               False
health_worker                  False
opinion_h1n1_vacc_effective    False
opinion_h1n1_risk              False
opinion_h1n1_sick_from_vacc    False
opinion_seas_vacc_effective    False
opinion_seas_risk              False
opinion_seas_sick_from_vacc    False
agegrp                         False
education_comp                 False
raceeth4_i                     False
sex_i                          False
inc_pov                        False
marital                        False
rent_own_r                     False
employment_status              False
census_region                  False
census_msa                     False
n_adult_r                      False
household_children             False
n_people_r                     False
employment_industry            False
employment_occupation          False
hhs_region                     False
state                          False
vacc_h1n1_f                    False
dtype: bool
'''

# 카디널리티 확인
train.describe(exclude='number').T.sort_values(by='unique')
```

![스크린샷 2021-08-17 20 36 34](https://user-images.githubusercontent.com/79494088/129719128-1d74793b-27a3-4290-82aa-77b3d49174c0.png)

## Feature Engineering

```py
# 특성공학
import numpy as np

def engineer(df):
    # 높은 카디널리티를 가지는 특성을 제거
    selected_cols = df.select_dtypes(include=['number', 'object'])
    labels = selected_cols.nunique() # 특성별 카디널리티 리스트
    selected_features = labels[labels <= 30].index.tolist() # 카디널리티가 30보다 작은 특성만 선택
    df = df[selected_features]
    
    # 새로운 특성을 생성
    behaviorals = [col for col in df.columns if 'behavioral' in col] 
    df['behaviorals'] = df[behaviorals].sum(axis=1)
    
    
    dels = [col for col in df.columns if ('employment' in col or 'seas' in col)]
    df.drop(columns=dels, inplace=True)
        
    return df


train = engineer(train)
val = engineer(val)
test = engineer(test)

# 피쳐에서 타겟 드롭
features = train.drop(columns=[target]).columns

# 훈련/검증/테스트 데이터를 특성과 타겟으로 분리
X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]
```

# 2️⃣ Modelling

```py
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 파이프라인 활용 로지스틱 회귀 모델링
pipe = make_pipeline(
    OneHotEncoder(), 
    SimpleImputer(), 
    StandardScaler(), 
    LogisticRegression(n_jobs=-1)
)
pipe.fit(X_train, y_train)

print('검증세트 정확도', pipe.score(X_val, y_val))

y_pred = pipe.predict(X_test)
'''
검증세트 정확도 0.8185268651405527
'''

pipe.named_steps
'''
{'logisticregression': LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=-1, penalty='l2',
                    random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                    warm_start=False),
 'onehotencoder': OneHotEncoder(cols=['opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
                     'opinion_h1n1_sick_from_vacc', 'agegrp', 'census_msa'],
               drop_invariant=False, handle_missing='value',
               handle_unknown='value', return_df=True, use_cat_names=False,
               verbose=0),
 'simpleimputer': SimpleImputer(add_indicator=False, copy=True, fill_value=None,
               missing_values=nan, strategy='mean', verbose=0),
 'standardscaler': StandardScaler(copy=True, with_mean=True, with_std=True)}
'''

# 시각화
import matplotlib.pyplot as plt

model_lr = pipe.named_steps['logisticregression']
enc = pipe.named_steps['onehotencoder']
encoded_columns = enc.transform(X_val).columns
coefficients = pd.Series(model_lr.coef_[0], encoded_columns)
plt.figure(figsize=(10,30))
coefficients.sort_values().plot.barh();
```

![스크린샷 2021-08-17 22 39 33](https://user-images.githubusercontent.com/79494088/129736022-be10ea70-1ac1-40fd-b5b7-990bdb8436dd.png)

## 모델 개발

### DecisionTree

```py
from sklearn.tree import DecisionTreeClassifier

pipe = make_pipeline(
    OneHotEncoder(use_cat_names=True),  
    SimpleImputer(), 
    DecisionTreeClassifier(random_state=1, criterion='entropy')
)

pipe.fit(X_train, y_train)
print('훈련 정확도: ', pipe.score(X_train, y_train))
print('검증 정확도: ', pipe.score(X_val, y_val))
'''
훈련 정확도:  0.9908667674880646
검증 정확도:  0.7572055509429486 
과적합 상태라고 볼 수 있다.
'''

# 시각화
import graphviz
from sklearn.tree import export_graphviz

model_dt = pipe.named_steps['decisiontreeclassifier']
enc = pipe.named_steps['onehotencoder']
encoded_columns = enc.transform(X_val).columns

dot_data = export_graphviz(model_dt
                          , max_depth=3
                          , feature_names=encoded_columns
                          , class_names=['no', 'yes']
                          , filled=True
                          , proportion=True)


display(graphviz.Source(dot_data))
```

![스크린샷 2021-08-17 22 45 26](https://user-images.githubusercontent.com/79494088/129737000-8362aae4-de92-48cb-b903-ca49fbc16342.png)

```py
from sklearn.metrics import f1_score

pipe = make_pipeline(
    OneHotEncoder(use_cat_names=True), 
    SimpleImputer(), 
    DecisionTreeClassifier(max_depth=7, random_state=2) # depth 변경
)

pipe.fit(X_train, y_train)
print('훈련 정확도', pipe.score(X_train, y_train))
print('검증 정확도', pipe.score(X_val, y_val))

# f1 score 계산
from sklearn.metrics import f1_score

pred = pipe.predict(X_val)
print('f1 스코어',f1_score(y_val, pred))
'''
훈련 정확도 0.8317468789846693
검증 정확도 0.8254062388803226
f1 스코어 0.551219512195122
'''

# imputer median 사용
pipe = make_pipeline(
    OneHotEncoder(use_cat_names=True), 
    SimpleImputer(strategy = 'median'), 
    DecisionTreeClassifier(max_depth=3, random_state=2) # depth 값 조절
)

pipe.fit(X_train, y_train)
print('훈련 정확도', pipe.score(X_train, y_train))
print('검증 정확도', pipe.score(X_val, y_val))
pred = pipe.predict(X_val)
print('f1 스코어',f1_score(y_val, pred))
'''
훈련 정확도 0.7994247249651573
검증 정확도 0.8054797770134029
f1 스코어 0.5866935483870968
'''

# 시각화
model_dt = pipe.named_steps['decisiontreeclassifier']

importances = pd.Series(model_dt.feature_importances_, encoded_columns)
plt.figure(figsize=(10,30))
importances.sort_values().plot.barh();
```

![스크린샷 2021-08-17 22 49 47](https://user-images.githubusercontent.com/79494088/129737707-b2a6d835-3f8a-4946-a1c6-66389dfe0e8f.png)

# 3️⃣ Kaggle Submit

```py
# 테스트 학습
tpred = pipe.predict(X_test)

# 제출 양식 생성
submission = pd.read_csv('submission.csv')
submission['vacc_h1n1_f'] = tpred
submission

# file export
submission.to_csv('submission.csv', index= False)
```

![스크린샷 2021-08-17 22 54 30](https://user-images.githubusercontent.com/79494088/129738559-701bb835-f7f0-4757-9337-7e75a869c8d5.png)

- 5번에 걸쳐 튜닝해 그래도 조금이나마 F1 점수를 올려서 기분이 좋았다.
- 처음 해본 캐글 제출인데... 정말이지... 너무 재밌다...
- 앞으로 배울 모델링 기법들이 기대된다.
- 하지만 지금까지 배운것들을 소홀히 하지말자!