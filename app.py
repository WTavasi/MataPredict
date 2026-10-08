import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MataPredict", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    columns = json.load(open("model_columns.json"))
    return model, columns


@st.cache_data
def load_data():
    return pd.read_csv("matatu_merged_clean.csv")


model, columns = load_model()
df = load_data()

st.title("MataPredict: will this matatu be overcrowded?")
tab_predict, tab_explore = st.tabs(["Predict a ride", "Explore the data"])

with tab_predict:
    left, right = st.columns(2)
    with left:
        origin = st.selectbox("Town of origin", sorted(df["travel_from"].unique()))
        car_type = st.selectbox("Vehicle type", sorted(df["car_type"].unique()))
        payment = st.selectbox("Payment method", sorted(df["payment_method"].unique()))
    with right:
        hour = st.slider("Hour of departure", 0, 23, 7)
        day = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday",
                                           "Friday", "Saturday", "Sunday"])
        month = st.selectbox("Month", sorted(df["month"].unique()))

    if st.button("Predict"):
        info = df[df["travel_from"] == origin].iloc[0]
        capacity = df.loc[df["car_type"] == car_type, "max_capacity"].iloc[0]
        ride = pd.DataFrame([{
            "route": info["route"],
            "travel_from": origin,
            "hour": hour,
            "minute": 0,
            "day_of_week": day,
            "month": month,
            "car_type": car_type,
            "payment_method": payment,
            "is_peak": int((5 <= hour <= 9) or (16 <= hour <= 20)),
            "max_capacity": capacity,
            "route_count": info["route_count"],
            "centroid_lat": info["centroid_lat"],
            "centroid_lon": info["centroid_lon"],
            "gis_missing": info["gis_missing"],
        }])
        row = pd.get_dummies(ride).reindex(columns=columns, fill_value=0)
        prob = model.predict_proba(row)[0, 1]
        st.metric("Chance of overcrowding", f"{prob:.0%}")
        if prob >= 0.5:
            st.error("This ride is likely to be overcrowded.")
        else:
            st.success("This ride is unlikely to be overcrowded.")

with tab_explore:
    st.subheader("Overcrowding rate by hour of day")
    st.bar_chart(df.groupby("hour")["is_overcrowded"].mean())
    st.subheader("Overcrowding rate by corridor")
    st.bar_chart(df.groupby("route")["is_overcrowded"].mean())
