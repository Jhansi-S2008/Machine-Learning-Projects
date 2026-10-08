# Mushroom Edibility Prediction

A Machine Learning web application that predicts whether a mushroom is **edible or poisonous** using a Decision Tree classification model.

## Project Overview

This project uses a Decision Tree machine learning algorithm to classify mushrooms based on their characteristics.

The application has a frontend where the user enters mushroom feature values. These values are sent to the backend through a REST API, where the trained Decision Tree model makes the prediction and returns the result to the frontend.

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- HTML
- CSS
- JavaScript
- REST API

## Machine Learning Model

The project uses a **Decision Tree Classifier** for mushroom classification.

The model is trained using the Mushroom dataset and learns patterns from the given mushroom features to predict the target class.

### Prediction Classes

- Edible
- Poisonous

## Project Workflow

```text
User Input
    ↓
Frontend
    ↓
REST API (/predict)
    ↓
Flask Backend
    ↓
Decision Tree Model
    ↓
Prediction
    ↓
Frontend Result
