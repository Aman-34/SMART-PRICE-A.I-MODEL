import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# ------------------ LOAD DATA ------------------
df = pd.read_csv("Cardetails_cleaned.csv", sep="\t")

# ------------------ CLEAN NUMERIC COLUMNS ------------------
df['mileage'] = pd.to_numeric(df['mileage'].astype(str).str.extract(r'([\d\.]+)')[0], errors='coerce')
df['engine'] = pd.to_numeric(df['engine'].astype(str).str.extract(r'([\d\.]+)')[0], errors='coerce')
df['max_power'] = pd.to_numeric(df['max_power'].astype(str).str.extract(r'([\d\.]+)')[0], errors='coerce')
df['km_driven'] = pd.to_numeric(df['km_driven'], errors='coerce')
df['seats'] = pd.to_numeric(df['seats'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')

df.dropna(subset=[
    'year','km_driven','mileage','engine',
    'max_power','seats','selling_price'
], inplace=True)

# ------------------ ENCODE CATEGORICAL FEATURES ------------------
df['name'] = df['name'].apply(lambda x: str(x).split(' ')[0].strip())

categorical_cols = ['name','fuel','seller_type','transmission','owner']
for col in categorical_cols:
    df[col] = df[col].astype(str)
    df[col] = df[col].factorize()[0]

# ------------------ FEATURES & TARGET ------------------
feature_cols = [
    'year','km_driven','mileage','engine',
    'max_power','seats','name','fuel',
    'seller_type','transmission','owner'
]

X = df[feature_cols]
y = df['selling_price']

# ------------------ TRAIN MODEL ------------------
model = LinearRegression()
model.fit(X, y)

# ------------------ SAVE MODEL (CORRECT WAY) ------------------
joblib.dump(model, "model.pkl")

print("✅ model.pkl created successfully (Streamlit-safe)")
