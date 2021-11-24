---
title: "[ML] Basic assumptions of linear regression data"
description: How to make good linear regression model? The four basic assumptions of linear regression analysis.
categories:
 - DS ENG SG
tags: [DS ENG SG, ML]
mathjax: enable
---

This posting is for preparing for the presentation of the Data Science English Study Group.

# How to make good linear regression model?

- To make a good model with linear regression analysis, the data must satisfy four basic assumptions.
- If the four basic assumptions are not satisfied, a proper linear regression model cannot be created.
- The four basic assumptions are,
    1. Linearity
    2. Independence
    3. Equal variance
    4. Normality

## Linearity

- Linearity is an important basic assumption in linear regression analysis.

![image](https://user-images.githubusercontent.com/79494088/142729659-74add53b-e231-4df0-940a-4a74e45be7a0.png)

- Looking at the table, it can be see that the variable that do not have a linear relationship with Sepal.Length is Sepal.Width.
- Making a linear regression model with this data, the P-Value is 0.152.
- Therefore, it has no influence.
- If some of the variables do not satisfy the linearity,
    1. Try adding another new variable.
    2. Try converting variables into logs, indices, and roots.
    3. Try removing variables that do not satisfy linearity.
    4. Force a linear regression model and pass the variable selection method.

## Independence
- It refers to a characteristic that has no correlation between independent variables.

![image](https://user-images.githubusercontent.com/79494088/142730224-4a407052-91d9-4dbf-96f2-e3f70996204a.png)

- Looking at the table, Force a variable with a high correlation.
- Although it was originally a significant variable, many similar variables occurred, resulting in insignificant results.
- This is multicollinearity. In Korean, '다중공선성'
- Variables that cause multicollinearity should be removed.

## Equal variance
- Equal variance is the same variance.
- The same variance means that it was evenly distributed without a specific pattern.

![image](https://user-images.githubusercontent.com/79494088/142730461-daf5454b-0f44-4b57-8846-75ab2883d24a.png)

- Looking at the table, I make weird ydata.
- As a result of regression analysis, there is no significant model.
- Let's look at the distribution of standardized residuals.

![image](https://user-images.githubusercontent.com/79494088/142730584-a0ebbf13-b4fa-4d69-bd84-6c43c295454e.png)

- The standardized residuals[스탠더다이즈드 리지주얼즈] does not satisfy the equal dispersibility and has a specific pattern with four lumps.
- So, important variables are not added to the analysis data and dropped.

## Nomality
- Normality means whether it has a normal distribution.

![image](https://user-images.githubusercontent.com/79494088/142979559-faf1ea1e-24c4-4118-9149-84b297480bf0.png)

- Looking at the table, i create a variable ydata that is concentrated on one side.
- With the hypothesis that "there is no difference from the normal distribution", the hypothesis is rejected because the p-value is 0.001.
- In order to satisfy normality, a similar method to solving equal variance is needed.


# Reference
- [선형 회귀분석의 4가지 기본가정](https://kkokkilkon.tistory.com/175?category=640119)