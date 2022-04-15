# Disaster Response Web App

An interactive deployable web app to analyze disaster messages into categories.

1. Installations and Instructions
2. Project Motivation
3. File Descriptions
4. How to Work with Project
5. Licensing, Authors, Acknowledgements, etc.


#1 
Welcome! To use this web app, first make sure to clone the repository.

`git clone repository`

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


#2 Project Motivation

The purpose of this project is to demonstrate a full data science workflow. This project is constructed to showcase the skills of pipeline creation using Python. It covers ETL pipelines, ML pipelines, and more specifically, NLP pipelines. All of the code is automated into one web app using Flask.

#3 File Descriptions

The workspace consists of 3 folders and the README containing the directions above. The `app` folder holds the template sub-folder, which holds all of the html for our web app. This can be customized for a nicer looking web page. Also within `app` is the run.py file, which holds all of the python code for running our app. This file is what should be modified if you want to add or change any of the visualization shown on the home page.

The `data` folder holds our data, as well as the process_data.py file. This file is executed to complete the ETL portion of our pipeline, where we ingest raw data, clean it, and save it back to a database in a clean format.

The final folder is named `models` and it holds our ML / NLP pipeline file named `train_classifier.py`. This file takes in cleaned data, prepares a ML pipeline, and saves a finalized and trained model back in a compressed format, which can be consumed by our web app.

The mini-readme found within the `workspace` folder holds general directions on how to get the project up and running as well.

#4 How to work with the Project

Follow the directions within the `workspace` folder to complete the subsequent steps.

#5  Licensing, Authors, Acknowledgements, etc.

All credit goes to the Udacity staff for providing learning material , resources, and starter code for this project.
