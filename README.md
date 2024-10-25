# Readme

This project started during an MIT Critical Data event in Tarragona (Spain). [Link](https://www.datathontarragona.com/)

The aim of the project was to create a model able to predict which patient admitted to the ICU would need CRRT therapy. The analysis is performed on two pubblicly available datasets:
  * Physionet MIMIC
  * Catalunia Dataset

To query the data you need to hava access to the datasets. Be sure to have an active subscription to Physionet and to the Tarragona open datasets. Create a ```.env``` file to store you credentials. Following is an example with the needed keys:
```
AWS_ACCESS_KEY_ID=xxxxxxxx
AWS_SECRET_ACCESS_KEY=yyyyyyyy
GOOGLE_CLOUD_PROJECT=zzzzzzzz
```

# How to

  * Run the ```get_data.py``` file to query the Tarragona dataset, and use the ```mimic_extraction.py``` to query the MIMIC dataset.
  * To run the analysis use the ```predictions.py``` script.

# Dependencies

Check the ```requirements.txt``` file