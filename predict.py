import joblib
import pandas as pd
MODEL_PATH = "linear_model.joblib"

model = joblib.load(MODEL_PATH)

# PUT YOUR FEATURES IN THE SAME ORDER AS TRAINING
sample = {
    "hours_studied": 5,
    "previous_scores": 90,
    "extracurricular_activities": 0,
    "sleep_hours": 5,
    "sample_question_papers_practiced": 7
}

X = pd.DataFrame([sample])
prediction = model.predict(X)[0]

print("Predicted performance_index:", prediction)
