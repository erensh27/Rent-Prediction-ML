import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("House_Rent_Dataset.csv")

# -----------------------------
# 2. Drop Unnecessary / Problematic Columns
# -----------------------------
df.drop(columns=['Posted On', 'Area Locality', 'Floor'], inplace=True)

# -----------------------------
# 3. One-Hot Encoding (SAFE COLUMNS ONLY)
# -----------------------------
df = pd.get_dummies(
    df,
    columns=[
        'City',
        'Area Type',
        'Furnishing Status',
        'Tenant Preferred',
        'Point of Contact'
    ],
    drop_first=True
)

# -----------------------------
# 4. Remove Extreme Rent Outliers
# -----------------------------
df = df[df['Rent'] < df['Rent'].quantile(0.95)]

# -----------------------------
# 5. Split Features & Target
# -----------------------------
X = df.drop('Rent', axis=1)
y = df['Rent']

# -----------------------------
# 6. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 7. Train Model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# 8. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------
# 9. Sample Prediction
# -----------------------------
sample = X.iloc[[0]]
print("Predicted Rent:", model.predict(sample)[0])
import joblib

joblib.dump(model, "rent_prediction_model.pkl")
print("Model saved successfully")