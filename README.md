# 🤖 Machine Learning App

The purpose of this application is to experience the process of creating predictive models easily in Python and scikit-learn.

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yh-machinelearning.streamlit.app/)

## Overview

The application guides users through the following steps:

1. **Data Loading and Display**:
   - The app loads the penguin dataset, separates the features (`X`) and target variable (`y`), and displays them.

2. **Data Visualization**:
   - A scatter plot visualizes the relationship between penguins' body mass and bill length, with points colored by species.

3. **User Input**:
   - A sidebar interface enables users to input specific features such as island, bill length, and sex for a penguin.

4. **Data Preparation**:
   - Categorical variables (e.g., island, sex) are encoded into numerical values. The new input data is then combined with the existing dataset.
   - The target variable (`species`) is also encoded into numerical labels.

5. **Model Training**:
   - A **Random Forest Classifier** is trained on the dataset.
   - The trained model predicts the penguin species based on the user’s input.

6. **Prediction Results Display**:
   - The predicted species and the associated probabilities are displayed.

## Machine Learning Method

- **Random Forest Classifier**:
  - The application uses a Random Forest Classifier, an ensemble learning method that employs multiple decision trees to perform classification. The final prediction is made by aggregating the predictions of individual trees, improving accuracy and handling noisy or non-linear data effectively.

