# Disaster Response Pipeline Project

[image1]: https://github.com/natsci-droid/Udacity_DS_P2_Disaster_Response/blob/main/figs/image1.png "App before text classification"
[image2]: https://github.com/natsci-droid/Udacity_DS_P2_Disaster_Response/blob/main/figs/image2.png "App with text classified"

### Introduction
This code is for Udacity's Disaster Response Project, under the [Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025). It analyses real disaster message data from Figure Eight to build a classification model.

There are 3 python scripts:
* process.data.py processes the data from csv files into a single sql database, with text messages cleaned ready for classification.
* train_classifier.py trains a random forest classifier to classify the messages on each of 36 categories. Grid search is used to find optimal parameters.
* run.py launches the web app for classification and viewing of the data.

Other files:
* The trained model is stored in the models folder as classifier.pkl.
* The processed data is stored in the data folder as DisasterResponse.db
* The input csv files, disaster_categories.csv and disaster_messages.csv are also stored in the data folder

### Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Screenshots

![App before text classification][image1]
![App with text classified][image2]