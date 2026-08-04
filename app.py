

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
import gradio as gr

df = pd.read_csv("diabetes.csv")
df = df.dropna(subset=["glyhb"]).copy()
df["Diabetes"] = (df["glyhb"] >= 6.5).astype(int)
df["BMI"] = 703 * df["weight"] / (df["height"] ** 2)
df = df.rename(columns={
    "stab.glu": "Glucose",
    "bp.1s": "BloodPressure",
    "age": "Age",
    "chol": "Cholesterol",
    "hdl": "HDL",
    "waist": "Waist",
    "hip": "Hip"
})

feature_cols = ["Glucose", "BloodPressure", "BMI", "Age",
                 "Cholesterol", "HDL", "Waist", "Hip"]
X = df[feature_cols].copy()
y = df["Diabetes"].copy()

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
)
model.fit(X_train, y_train)

medians = X_train.median()


def predict_diabetes(glucose, blood_pressure, bmi, age):
    new_patient = pd.DataFrame([{
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "BMI": bmi,
        "Age": age,
        "Cholesterol": medians["Cholesterol"],
        "HDL": medians["HDL"],
        "Waist": medians["Waist"],
        "Hip": medians["Hip"],
    }])
    pred = model.predict(new_patient)[0]
    proba = model.predict_proba(new_patient)[0][1]
    result = "Diabetes" if pred == 1 else "No Diabetes"
    return f"Prediction: {result}  (probability: {proba:.1%})"


demo = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="BMI"),
        gr.Number(label="Age"),
    ],
    outputs="text",
    title="Diabetes Prediction (Gradient Boosting)",
    description="Enter patient info to predict diabetes risk."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
