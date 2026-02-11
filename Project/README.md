In this project my goal is to do exercise classification from images and videos.

There are two Jupyter Notebooks, one for the image classification task and one for the video classification task. The notebooks are meant to be run from Google Colab and haven't been tested locally.

In them the whole process of creating the models and experimentation is described. 

The results are found in two Excel files called: model_report_image_classification.xlsx
and model_report_video_classification.xlsx

A script that demonstrates how the best model works is found in the file application_demo.py

Future work includes: 
- Creating a model that determines when the exercise starts and stops to select better frames on which to classify the exercise and reduce noise;
- Creating a model that takes as input the video and the exercise and returns as output the number of repetitions completed in the video;
- Creating a model that takes in the video and exercise and returns as output commentary if you are doing the exercise with correct form.
