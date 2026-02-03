import joblib

lr = joblib.load("models/lr.pkl")
svm = joblib.load("models/svm.pkl")
rf = joblib.load("models/rf.pkl")
scaler = joblib.load("models/scaler.pkl")
