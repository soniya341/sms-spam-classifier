import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load the messages
df = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 2. Split into practice and test messages
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42)

# 3. Turn words into numbers
vec = TfidfVectorizer(stop_words="english")
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# 4. First model: Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("Naive Bayes")
print(classification_report(y_test, model.predict(X_test_vec)))

# 5. Second model: Logistic Regression
model2 = LogisticRegression(max_iter=1000, class_weight="balanced")
model2.fit(X_train_vec, y_train)
print("Logistic Regression")
print(classification_report(y_test, model2.predict(X_test_vec)))

# 6. Save the winner
joblib.dump(model2, "data/spam_model.pkl")
joblib.dump(vec, "data/vectorizer.pkl")
print("Model saved!")

# 7. Try our own messages
tests = [
    "You won a free prize! Click now to claim",
    "Hey, are we still meeting for lunch today?",
    "URGENT! Your account is locked. Send your password",
]
for t in tests:
    result = model2.predict(vec.transform([t]))[0]
    print(t, "->", "SPAM" if result == 1 else "NOT SPAM")

