# Udacity Data Scientist Project 2 - Disaster Response Pipeline

### Summary
The purpose of this project is to help organization to categorize text messages so they know if and what kind of help is needed. \
A model is trained using messages and their categories. The performance of the model is evaluted in classification_report. \
The model is then used to predict the categories of new messages which the user inserts into a website. \
Besides this functionality the website shows some graphics of the underlying dataset which was available for training and testing the model. 

### Table of content

--- app\
------ templates\
---------- go.html\
---------- master.html\
------ run.py\
--- data\
------    DisasterResponse.db\
------    disaster_categories.csv\
------    disaster_messages.csv\
------    process_data.py\
--- models\
------    classifier.pkl\
------    train_classifier.py\
-- classification_report.csv

### File Description
1.1 app/templates/go.html: part of the website\
1.2 app/templates/master.html: part of the website\
1.3 app/run.py: Builds and runs the website

2.1 data/DisasterResponse.db: includes data used to build the model, output of process_data.py, input for train_classifier.py\
2.2 data/disaster_categories.csv: raw data, input for process_data.py\
2.3 data/disaster_messages.csv: raw data, input for process_data.py\
2.4 data/process_data.py: Extracts, transforms input data and outputs the DB.

3.1 models/classifier.pkl: final model, output of train_classifier.py\
3.2 models/train_classifier.py: Builds the model and ouptuts the model. 

4 classification_report.csv: result of the model evaluation.


### How to run the programm
1. Create a processed sqlite db:\
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db\
2. Train and save a pkl model:\
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl\
3. Deploy the application locally:\
app/python run.py
