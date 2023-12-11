# Boeing SDR Classifier package
This is a package to classify SDR records for Depressurization event.  

## Build the whl file
Run the command in the root folder to create the whl file in the _dist_ folder

`python setup.py bdist_wheel`

## Virtual Environment
It is highly recommended to use venv, virtualenv or conda python environments. Read more about creating virtual environments via venv

https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments

## Install boeing_sdr_classifier and its dependencies
pip install boeing_sdr_classifier-0.0.1-py3-none-any.whl

## Example
Follow the code snippet below to test and call the prediction method from the Depressurization model

```
from boeing_sdr_classifier import boeing_sdr_api
import pandas as pd

my_model = boeing_sdr_api.BoeingSDRAPI()

#test the prediction method
my_model.test_sdr_depressurization_predictions()

df = pd.read_csv('../sdr_gold_labels.csv')
records = df["NormalizedEventText"]
#pass in a record list for prediction
pred, probs = my_model.get_depressurization_predictions(records)

df['Prediction'] = pred
df['Prob'] = probs

print(df.head(2))

print("Done")
```
