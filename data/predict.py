import joblib

model = joblib.load("data/spam_model.pkl")
vec = joblib.load("data/vectorizer.pkl")

print("Type a message. Type q to quit.")
while True:
    text = input("Message: ")
    if text.lower() == "q":
        break
    result = model.predict(vec.transform([text]))[0]
    print("-> SPAM" if result == 1 else "-> NOT SPAM")

