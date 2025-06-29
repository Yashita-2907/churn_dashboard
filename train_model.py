import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle

df = pd.read_csv("C:/Users/Oman Air/Documents/churn-dashboard/data/churn.csv")
df.dropna(inplace=True)
df.drop(["customerID"], axis=1, inplace=True)

# Encode categorical
for col in df.select_dtypes("object").columns:
    if df[col].nunique() <= 2:
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col])

X = df.drop("Churn", axis=1)
y = LabelEncoder().fit_transform(df["Churn"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

import joblib

# After training and saving your model:
joblib.dump(model, "model.pkl")

# Save the features the model was trained on:
model_features = X.columns.tolist()  # Replace X with the DataFrame used for training
joblib.dump(model_features, "model_features.pkl")
