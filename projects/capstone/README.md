# Machine Learning Engineer Nanodegree
## Specializations
## Project: Capstone Proposal and Capstone Project

**Note**

The Capstone is a two-staged project. The first is the proposal component, where you can receive valuable feedback about your project idea, design, and proposed solution. This must be completed prior to your implementation and submitting for the capstone project. 

You can find the [capstone proposal rubric here](https://review.udacity.com/#!/rubrics/410/view), and the [capstone project rubric here](https://review.udacity.com/#!/rubrics/108/view). Please ensure that you are following directions correctly before submitting these two stages which encapsulate your capstone.

Please email [machine-support@udacity.com](mailto:machine-support@udacity.com) if you have any questions.

**Python version**: 3.6.6
## Directory file contents

-	train/train.csv – tabular data containing profile features and the adoption speed target variable.
-	test/test.csv – tabular data containing profile features
-	test_sentiment/ - contains training metada generate from description processing
-	test_sentiment/ - contains testing metada generate from description processing
-	breed_labels –PetID and breed of pet
-	color_labels – PetID and color of pet
-	state_labels – PetID state location state

### How to get the data?

By cloning this repository all required data files will be downloaded. However, a second option to get all files will be:
1.	Go to https://www.kaggle.com/c/petfinder-adoption-prediction/data
2.	Scroll down to the Data Sources section
3.	Download each required file mentioned above.

**Note** that it is important to maintain the file structure present in this repository because the code file has paths according to this structure.

### Required python libraries
* Numpy
* Pandas
* Matplotlib
* Seaborn
* scikit-learn
* json
* pprint
* xgboost
* lightgbm

## Code File
File “MLND - Capstone Code.ipynb” contains all code for this project. It contains several sections described below.
### Visualize plots instructions
The visualization section will display all plots if run automatically. If plotting is not  wanted feel free to comment these section it will not interfere with the modeling section.
### Modeling section
By default, the Random Forest and LightGBM implementations is commented. XGBoost code is ready to run. **Each modeling cell ends with an output variable, if all models are run sequentially this file will be overwritten by the results of LightGBM**.
### Generate solution file
The last cell contains code to generate a csv file. This file can be uploaded to Kaggle for evaluation.
