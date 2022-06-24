# Tabular Lesion Prediction

## Overview
This is work in progress, mostly to exhibit some coding style and some ideas.

I will not provide information on how the datasets are created, since I am under a <span style="color: red">no-disclosure</span> agreement,
but the values are extrapolated from the ProstateX dataset that is available online.

<font color='blue'>_A CI/CD pipeline has been implemented via tox and Git Action_</font>

The data folder contains six datasets I will be working on.

## Dependencies

- mlxtend
- shap
- scikit-learn-intelex
- pytorch-tabular
- xgboost
- imbalanced-learn
- mitosheet
- sklearn-genetic
- flake8
- tox
- pytest
- pytest-cov
- mypy
- kaleido
- yellowbrick

## Task

- [x] Project Setup
  - [x] Project Structure
  - [x] Test Implementation
  - [x] Git Action connection
- [] Classifier Implementation
  - [x] LogisticRegression Classifier
  - [x] XGBClassifier
  - [x] RidgeClassifier
  - [] PassiveAggressiveClassifier
  - [] SGDOneClassSVM
  - [] SGDClassifier
  - [] Pytorch Tabular
- [x] Data Preprocessing
  - [x] Delete all zero Features
  - [x] Delete Features with STD larger than 10000
  - [x] Delete Features with low variance (variance threshold 0.005)
- [x] Feature Selection
  - [x] Genetic Algorithm Approach

  
All constructive comments are appreciated and valued.

![Tests](https://github.com/fabiogeraci/tabular_lesion/actions/workflows/tests.yml/badge.svg)