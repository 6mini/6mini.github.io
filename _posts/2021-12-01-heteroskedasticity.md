---
title: "[ML] Heteroscedasticity"
description: Problems and solutions of heteroscedasticity.
categories:
 - TADA English Study
tags: []
mathjax: enable
---

This posting is for preparing for the presentation of the Data Science English Study Group.

# Heteroscedasticity
- This means that the variance is different.
- In other to, it means that the standard error of the regression coefficient is different.
- The t-value is required to determine the significance of the regression coefficient.
    - T-Value: The regression coefficient divided by the standard error.

![image](https://user-images.githubusercontent.com/79494088/143775458-768b8885-5288-4d8c-a5c0-5a9227a6026f.png)

- The distribution of points in the table is the standard error of the regression coefficient.
- I don't know which part of the standard error to use because the variance is not constant.
- As x increases in the graph, y also increases.
- And the standard error increases.
- In order to, the standard error can be expressed as a function of the independent variable.

![image](https://user-images.githubusercontent.com/79494088/143775724-c048c226-07e3-4e6a-893f-e49f4e3affc8.png)

- If the residual degree is the pattern of this graph, the model is heteroscedastic.

## Problem
- If all the basic assumptions in the regression model are met,
- It has the characteristics of BLUE.
- BLUE are,

1. unbiasedness
2. Linearity
3. Consistency
4. Efficiency

- But if there's a heteroscedasticity,
- The variance of the estimator increases.
- Therefore, it cannot have the characteristics of BLUE because it does not have the efficiency of having a minimum variance.

## How to check
- The way to check the heteroscedasticity are,

1. Scatter Plot
2. Residual Plot
3. White Test
4. Goldfeld Quandt test

## Solutions

### Robust Standard Error
- This is a way to be recognized as a solution to stability and heteroscedasticity.

### Weight Least Square Regression법)
- It is a method of finding a function of heteroscedasticity, creating and adding an independent variable with its inverse function.
- It is theoretically easy but realistically difficult.

### GLS/FGLS Regression
- This is a generalized least squares method.
- This is fundamentally similar to WLS.
- This is also theoretically easy but realistically difficult.