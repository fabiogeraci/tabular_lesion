# Test & License

![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)

![Tests](https://github.com/fabiogeraci/tabular_lesion/actions/workflows/tests.yml/badge.svg)

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
- pytest~=6.2.5
- pytest-cov>=3.0.0
- mypy>=0.961
- kaleido>=0.2.1
- yellowbrick>=1.4
- skl2onnx>=1.10
- onnxmltools>=1.10
- onnxruntime>=1.10
- pytest-pythonpath>=0.7.4
- pandas~=1.1.5
- plotly~=5.9.0
- numpy~=1.22.4
- scikit-learn~=1.1.1
- scikit-optimize~=0.9.0

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
   - [X]  Genetic Algorithm Approach (by sklearn-genetic)
- [] Classifier Implementation
  - [X]  LogisticRegression Classifier
  - [X]  XGBClassifier
  - [X]  RidgeClassifier
  - [X]  GaussianNB Classifier
  - [] PassiveAggressiveClassifier
  - [] SGDOneClassSVM
  - [] SGDClassifier
  - [] Pytorch Tabular
- [] ONNX implementation (examples of class inheritance)
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

```
cd src/classifier
```

At the moment, the following classifier are available by running one of the following commands:

```
python logistic_classification.py
python ridge_classification.py
python xgboost_classification.py
python gaussiannb_classification.py
```

# <img height="20" width="20" src="https://github.githubassets.com/images/icons/emoji/unicode/1f6e0.png">Skills

## Deep Learning Frameworks
|<img align="center" src="https://camo.githubusercontent.com/e0af84521a474956fc781af46a392cede22b59034415df6d4a876ce55c7f2696/68747470733a2f2f65787465726e616c2d636f6e74656e742e6475636b6475636b676f2e636f6d2f69752f3f753d687474707325334125324625324669312e77702e636f6d25324664617461736369656e63657765656b2e6f726725324677702d636f6e74656e7425324675706c6f616473253246323031392532463132253246666173742e61695f2e6a706725334673736c2533443126663d31266e6f66623d31" height="40" data-canonical-src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi1.wp.com%2Fdatascienceweek.org%2Fwp-content%2Fuploads%2F2019%2F12%2Ffast.ai_.jpg%3Fssl%3D1&amp;f=1&amp;nofb=1" style="max-width: 100%;">|<img align="center" src="https://raw.githubusercontent.com/valohai/ml-logos/master/keras-text.svg" height="35" style="max-width: 100%;">|<img align="center" src="https://raw.githubusercontent.com/valohai/ml-logos/master/pytorch.svg" height="35" style="max-width: 100%;">|




## Hyperparameter Optimization
<img align="center" src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" height="35" style="max-width: 100%;">

## Experiment Management
<img align="center" src="https://raw.githubusercontent.com/wandb/client/master/.github/wb-logo-lightbg.png" height="35" style="max-width: 100%;">
<img align="center" src="https://camo.githubusercontent.com/7c78ff7d75128f118c43dc696967feaa317ae7e6c3fe2628f4ff9f51ebd26e9e/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f736974652d6173736574732f696d616765732f70726f6a6563742d6c6f676f732f74656e736f72626f6172642d6c6f676f2d736f6369616c2e706e67" height="35" data-canonical-src="https://www.tensorflow.org/site-assets/images/project-logos/tensorboard-logo-social.png" style="max-width: 100%;">

## Model Deployment
<img align="center" src="https://raw.githubusercontent.com/onnx/onnx/main/docs/ONNX_logo_main.png" height="35" style="max-width: 100%;">

## Hardware
<img align="center" src="https://camo.githubusercontent.com/6e984be00dd03cf86490134cdc29c38550f470720f006cbff657d1956e30c3ff/68747470733a2f2f706e67696d672e636f6d2f75706c6f6164732f696e74656c2f696e74656c5f504e4731322e706e67" height="35" data-canonical-src="https://pngimg.com/uploads/intel/intel_PNG12.png" style="max-width: 100%;">
<img align="center" src="https://img.shields.io/badge/Heroku-430098?style=for-the-badge&logo=heroku&logoColor=white">
<img align="center" src="https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white">
<img align="center" src="https://img.shields.io/badge/Amazon_AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white">

## Software Engineering
<img align="center" src="https://raw.githubusercontent.com/dnth/logos/master/logos/git.svg" height="35" style="max-width: 100%">
<img align="center" src="https://raw.githubusercontent.com/dnth/logos/master/logos/jupyter.svg" height="35" style="max-width: 100%;">
<img align="center" src="https://camo.githubusercontent.com/d9f06d243f246d645f89946ea6b0539841c070d849a1fdd7acdab235cf794236/68747470733a2f2f73332d75732d776573742d312e616d617a6f6e6177732e636f6d2f756d6272656c6c612d626c6f672d75706c6f6164732f77702d636f6e74656e742f75706c6f6164732f323031362f30332f646f636b65722d6c6f676f2e6a7067" height="35" data-canonical-src="https://s3-us-west-1.amazonaws.com/umbrella-blog-uploads/wp-content/uploads/2016/03/docker-logo.jpg" style="max-width: 100%;">
<img align="center" src="https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white">
<img align="center" src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252">
