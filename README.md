# Tabular Lesion Prediction

## Overview

This is work in progress, mostly to exhibit some coding style and some ideas.

I will not provide information on how the datasets are created, since I am under a _**no-disclosure agreement**_,
but the values are extrapolated from the ProstateX dataset that is available online.

**_A CI/CD pipeline has been implemented via tox and Git Action_**

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

- [X]  Project Setup
  - [X]  Project Structure
  - [X]  Test Implementation
  - [X]  Git Action connection

- [] Classifier Implementation

  - [X]  LogisticRegression Classifier
  - [X]  XGBClassifier
  - [X]  RidgeClassifier

  - [] PassiveAggressiveClassifier
  - [] SGDOneClassSVM
  - [] SGDClassifier
  - [] Pytorch Tabular

- [X]  Data Preprocessing
  - [X]  Delete all zero Features
  - [X]  Delete Features with STD larger than 10000
  - [X]  Delete Features with low variance (variance threshold 0.005)
- [X]  Feature Selection
  - [X]  Genetic Algorithm Approach

All constructive comments are appreciated and valued.

## Badges

![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)

![Tests](https://github.com/fabiogeraci/tabular_lesion/actions/workflows/tests.yml/badge.svg)
