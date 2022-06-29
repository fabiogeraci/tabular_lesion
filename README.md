# Tabular Lesion Prediction

## Overview

This is work in progress, mostly to exhibit some coding style and some ideas.

I will not provide information on how the datasets are created, since I am under a _**no-disclosure agreement**_,
but the values are extrapolated from the ProstateX dataset that is available online.

**_A CI/CD pipeline has been implemented via tox and Git Action_**

The data folder contains six datasets I will be working on.

## Dependencies

- mlxtend>=0.20.0
- shap>=0.41.0
- scikit-learn-intelex>=2021.6.3
- pytorch-tabular>=0.7.0
- xgboost>=1.6.1
- imbalanced-learn>=0.9.1
- mitosheet>=0.1.432
- sklearn-genetic>=0.5.1
- flake8>=4.0.1
- tox>=3.25.0
- pytest>=3.25.0
- pytest-cov>=3.0.0
- mypy>=0.961
- kaleido>=0.2.1
- yellowbrick>=1.4
- skl2onnx>=1.10
- onnxmltools>=1.10
- onnxruntime>=1.10

## Task

- [X]  Project Setup
- [X]  Project Structure
- [X]  Test Implementation
- [X]  Git Action connection
- [X]  Data Preprocessing
   - [X]  Delete all zero Features
   - [X]  Delete Features with STD larger than 10000
   - [X]  Delete Features with low variance (variance threshold 0.005)
   - [X]  Oversampling via SMOTE
- [X]  Feature Selection
   - [X]  Genetic Algorithm Approach
- [] Classifier Implementation
  - [X]  LogisticRegression Classifier
  - [X]  XGBClassifier
  - [X]  RidgeClassifier
  - [] PassiveAggressiveClassifier
  - [] SGDOneClassSVM
  - [] SGDClassifier
  - [] Pytorch Tabular
- [] ONNX implementation (example class inheritance with classmethod)
  - [x] Save sklearn best estimator to ONNX
  - [x] Save xgboost model to ONNX
  - [] Inference via ONNX
  
_**All constructive comments are appreciated and valued.**_

## Installation

```
git clone https://github.com/fabiogeraci/tabular_lesion.git
pip install -r requirements.txt 
```

## Usage

To be added

## Badges

![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)

![Tests](https://github.com/fabiogeraci/tabular_lesion/actions/workflows/tests.yml/badge.svg)
