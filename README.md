## Service Difficult Report (SDR) classifier package
:rocket: This is the source code for the SDR Classifier package that classifies potential aviation safety hazards from textual data.  The work is a collaboration between FAA and Boeing data scientist teams

**Version 0.0.1 out now!**
[Check out the release notes here.](https://github.com/Boeing/sdr-hazards-classification/releases)

[![tests](https://github.com/Boeing/sdr-hazards-classification/actions/workflows/python-package.yml/badge.svg)](https://github.com/Boeing/sdr-hazards-classification/actions/workflows/python-package.yml)
[![Current Release Version](https://img.shields.io/github/release/Boeing/sdr-hazards-classification.svg?style=flat-square&logo=github)](https://github.com/Boeing/sdr-hazards-classification/releases)
[![pypi Version](https://img.shields.io/pypi/v/sdr-classifier.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/sdr-classifier/)
[![Python wheels](https://img.shields.io/badge/wheels-%E2%9C%93-4c1.svg?longCache=true&style=flat-square&logo=python&logoColor=white)](https://github.com/Boeing/sdr-hazards-classification/releases)
[![PyPi downloads](https://static.pepy.tech/personalized-badge/sdr-classifier?period=total&units=international_system&left_color=grey&right_color=orange&left_text=pip%20downloads)](https://pypi.org/project/sdr-classifier/)
[![Boeing on Twitter](https://img.shields.io/twitter/follow/boeing.svg?style=social&label=Follow)](https://twitter.com/boeing)

## Demo
![](https://github.com/Boeing/sdr-hazards-classification/blob/hai-branch/img/sdr_classifier.gif)

## Virtual Environment
It is highly recommended to use venv, virtualenv or conda python environments. Read more about creating virtual environments via venv
https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments

## Build and pip install the whl file locally
Run the command in the root folder to create the whl file in the _dist_ folder
```
git clone https://github.com/Boeing/sdr-hazards-classification
python setup.py bdist_wheel
pip install ./distr/sdr_classifier-0.0.1-py3-none-any.whl
```

## Install sdr_classifier from Pypi with pip
```
pip install sdr_classifier
```

## Example
:airplane: Follow the code snippet below to test and call the prediction method from the Depressurization model

```
from sdr_classifier import sdr_api
import pandas as pd

my_model = sdr_api.SdrInferenceAPI()

#test the prediction method
my_model.test_sdr_depressurization_predictions()

df = pd.read_csv('./src/sdr_classifier/data/SDR_Example.csv')
records = df["Text"]
#pass in a record list for prediction
pred, probs = my_model.get_predictions(records)

df['Prediction'] = pred
df['Prob'] = probs

print(df.head(2))

print("Done")
```
