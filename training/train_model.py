import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from feature_utils import extract_features

DATA_DIR = "data"
X, y = [], []

for label, cls in enumerate(["human", "ai"]):
    folder = os.path.join(DATA_DIR, cls)
    for file in os.listdir(folder):
        try:
            feats = extract_features(os.path.join(folder, file))
            X.append(feats)
            y.append(label)
        except:
            continue

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

lr = CalibratedClassifierCV(
    LogisticRegression(max_iter=300), method="sigmoid"
)
svm = CalibratedClassifierCV(
    SVC(kernel="rbf", probability=True), method="sigmoid"
)
rf = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=300), method="sigmoid"
)

lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
rf.fit(X_train, y_train)

os.makedirs("../models", exist_ok=True)

joblib.dump(lr, "../models/lr.pkl")
joblib.dump(svm, "../models/svm.pkl")
joblib.dump(rf, "../models/rf.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("Models trained & saved successfully")
