# Service Difficult Report (SDR) classifier package
This is the source code for SDR Classifier package that classifies SDR records for potential safety hazards.  The work is a collaboration between FAA and Boeing 

## Demo
![](https://github.com/Boeing/sdr-hazards-classification/blob/hai-branch/img/sdr_classifier.gif)

## Virtual Environment
It is highly recommended to use venv, virtualenv or conda python environments. Read more about creating virtual environments via venv
https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments

## Build the whl file and pip install whl file 
Run the command in the root folder to create the whl file in the _dist_ folder
```
python setup.py bdist_wheel
pip install sdr_classifier-0.0.2-py3-none-any.whl
```

## Install sdr_classifier with pip
```
pip install sdr_classifier
```

## Example
Follow the code snippet below to test and call the prediction method from the Depressurization model

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
