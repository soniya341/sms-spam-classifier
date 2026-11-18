# SMS Spam Classifier

A machine learning project that decides if an SMS message is spam or not.

## How it works
1. Loads the SMS Spam Collection dataset (5,572 messages)
2. Turns words into numbers with TF-IDF
3. Trains two models: Naive Bayes and Logistic Regression
4. Saves the better model (Logistic Regression)

## Results
| Model | Accuracy | Spam recall |
|---|---|---|
| Naive Bayes | 0.97 | 0.75 |
| Logistic Regression | 0.98 | 0.91 |

## How to run
pip install pandas scikit-learn joblib
python data/train.py
python data/predict.py

## Dataset
SMS Spam Collection Dataset from Kaggle (UCI Machine Learning Repository).