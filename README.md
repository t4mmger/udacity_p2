# Udacity Data Scientist Project 2 - Disaster Response Pipeline

### Summary
The purpose of this project is to help organization to categorize text messages so they know if and what kind of help is needed. \
A model is trained using messages and their categories. The performance of the model is evaluted in classification_report. \
The model is then used to predict the categories of new messages which the user inserts into a website. \
Besides this functionality the website shows some graphics of the underlying dataset which was available for training and testing the model. 

### Table of content

--- app\
------ templates\
------ go.html\
------ master.html\
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

### How to run the programm
1. Create a processed sqlite db:\
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db\
2. Train and save a pkl model:\
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl\
3. Deploy the application locally:\
python run.py
