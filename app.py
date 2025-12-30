import pandas as pd
import streamlit as st
import joblib
import os

# ------------------ LOAD MODEL ------------------
model_path = os.path.join(os.getcwd(), "model.pkl")
model = joblib.load(model_path)

st.title("🚗 SMART PRICE A.I MODEL")

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
    'year','km_driven','mileage',
    'engine','max_power','seats'
], inplace=True)

# ------------------ PROCESS BRAND ------------------
df['name'] = df['name'].apply(lambda x: str(x).split(' ')[0].strip())

# ------------------ CREATE CATEGORY MAPS ------------------
cat_cols = ['name','fuel','seller_type','transmission','owner']
cat_maps = {}

for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col], uniques = pd.factorize(df[col])
    cat_maps[col] = dict(zip(uniques, range(len(uniques))))

# ------------------ USER INPUT ------------------
name = st.selectbox("Car Brand", cat_maps['name'].keys())
fuel = st.selectbox("Fuel Type", cat_maps['fuel'].keys())
seller_type = st.selectbox("Seller Type", cat_maps['seller_type'].keys())
transmission = st.selectbox("Transmission Type", cat_maps['transmission'].keys())
owner = st.selectbox("Owner Type", cat_maps['owner'].keys())

year = st.slider("Year of Manufacture",
                 int(df['year'].min()),
                 int(df['year'].max()),
                 int(df['year'].median()))

km_driven = st.slider("Kilometers Driven",
                      int(df['km_driven'].min()),
                      int(df['km_driven'].max()),
                      int(df['km_driven'].median()))

mileage = st.number_input("Mileage (kmpl)",
                          float(df['mileage'].min()),
                          float(df['mileage'].max()),
                          float(df['mileage'].median()))

engine = st.number_input("Engine CC",
                         float(df['engine'].min()),
                         float(df['engine'].max()),
                         float(df['engine'].median()))

max_power = st.number_input("Max Power (BHP)",
                            float(df['max_power'].min()),
                            float(df['max_power'].max()),
                            float(df['max_power'].median()))

seats = st.slider("Number of Seats",
                  int(df['seats'].min()),
                  int(df['seats'].max()),
                  int(df['seats'].median()))

# ------------------ PREDICTION ------------------
if st.button("Predict Price"):
    input_df = pd.DataFrame([[
        year,
        km_driven,
        mileage,
        engine,
        max_power,
        seats,
        cat_maps['name'][name],
        cat_maps['fuel'][fuel],
        cat_maps['seller_type'][seller_type],
        cat_maps['transmission'][transmission],
        cat_maps['owner'][owner]
    ]], columns=[
        'year','km_driven','mileage','engine',
        'max_power','seats','name','fuel',
        'seller_type','transmission','owner'
    ])

    price = model.predict(input_df)[0]
    price = max(10000, price)

    st.success(f"💰 Predicted Selling Price: ₹{price:,.0f}")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown(
    "Created by **AMAN JOSHI**  "
    "(CO-Powered by **IIT Genius Web**)"
)
