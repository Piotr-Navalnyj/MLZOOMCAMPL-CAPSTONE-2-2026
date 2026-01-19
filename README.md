# MLZOOMCAMPL-CAPSTONE-2-2026

# Student Performance Prediction

This project predicts a student's **Performance Index** based on study habits and related factors using **Linear Regression**.

The goal of this project is to predict student performance based on study habits and related factors, and to evaluate how accurately these factors can explain academic outcomes.

---

## Problem Description

Student performance is influenced by multiple factors such as:
- hours studied
- previous scores
- sleep habits
- extracurricular activities
- practice with sample questions

The objective of this project is to predict a numerical **performance index** using these features and evaluate how accurately a machine learning model can capture these relationships.

---

## Dataset

The dataset contains the following features:

- `hours_studied`
- `previous_scores`
- `extracurricular_activities` (Yes / No)
- `sleep_hours`
- `sample_question_papers_practiced`

Target variable:
- `performance_index`

The dataset is provided as a CSV file.

---

## Exploratory Data Analysis (EDA)

EDA was performed in the notebook `Capstone2.ipynb` and includes:
- dataset structure and summary statistics
- missing value analysis
- feature distributions
- correlation analysis
- relationship between features and the target variable

Key findings:
- Hours studied and previous scores have strong positive correlation with performance.
- Sleep hours show a moderate effect.
- Extracurricular activities have a smaller but noticeable impact.

---

## Model Training

Three models were trained and evaluated:
- Linear Regression
- Random Forest
- XGBoost

Models were compared using **RMSE** and **R²** on validation and test sets.

### Final Model
**Linear Regression** was selected as the final model because:
- it achieved the lowest test RMSE
- validation and test errors were consistent
- the model is simple and interpretable

---

## Evaluation Metrics

The following metrics were used:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

The final model achieved stable performance with low prediction error on unseen data.

---

## Explanation on how to use this model

1: Download all stuff in requirenment.txt most of them you can easily be downloaded by typed in cmd: pip install "all things from requirenments"

step 2: download data StudentPerformance.csv

step 3: open jupyter notebook

step 4: copy file Train and paste in first cell of your notebook or just download two files train.py and predict.py and run them

step 5: change directory from where you will open data

step 6: run it

step 7: copy file predic.py and paste it into second cell of your juoyter notebook

step 8: change directory from where you will open data

step 9: run it and it should show the result you can also change feauters for this model (its explained in file)

!!!dont forget to change "data.csv" to your real file name and directory!!!

