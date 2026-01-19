import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



DATA_PATH = r""
TARGET = "performance_index"



df = pd.read_csv(DATA_PATH, encoding="latin1")


df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")


if "extracurricular_activities" in df.columns:
    df["extracurricular_activities"] = (
        df["extracurricular_activities"]
        .astype(str)
        .str.lower()
        .map({"yes": 1, "no": 0})
    )


df = df.dropna()



n = len(df)
np.random.seed(42)
idx = np.random.permutation(n)

n_train = int(0.6 * n)
n_val = int(0.2 * n)

train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

df_train = df.iloc[train_idx]
df_val = df.iloc[val_idx]
df_test = df.iloc[test_idx]



X_train = df_train.drop(TARGET, axis=1).values
X_val = df_val.drop(TARGET, axis=1).values
X_test = df_test.drop(TARGET, axis=1).values

y_train = df_train[TARGET].values
y_val = df_val[TARGET].values
y_test = df_test[TARGET].values



model = LinearRegression()
model.fit(X_train, y_train)



def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

rmse_val, mae_val, r2_val = evaluate(y_val, val_pred)
rmse_test, mae_test, r2_test = evaluate(y_test, test_pred)



print("\n===== VALIDATION =====")
print("RMSE:", rmse_val)
print("MAE :", mae_val)
print("R2  :", r2_val)

print("\n===== TEST =====")
print("RMSE:", rmse_test)
print("MAE :", mae_test)
print("R2  :", r2_test)



joblib.dump(model, "linear_model.joblib")
print("\nModel saved as linear_model.joblib")
